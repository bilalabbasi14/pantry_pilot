"""Bounded single-agent loop for the Recipe & Grocery Planner.

Each step: assemble layered context -> one model decision -> schema validation -> semantic
validation/authorization -> (optional) allow-listed tool via the gateway -> state update.
The application, not the model, owns validation, retries, budgets and stopping.
"""
import asyncio
import json
import logging
import re
import time
from datetime import date
from typing import Any
from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import ValidationError
from app.config import settings
from app.grocery import load_catalog, render_plan
from app.llm import LLMError, ModelClient, estimate_cost, get_client
from app.models import AgentDecision, AgentState, ArenaRequest, ArenaResponse, Metrics, PlanSandbox, StepRecord, ToolTrace
from app.policy import GroundingText, detect_boundary_requests, leaks_system_prompt, validate_semantics
from app.prompts import CANARY, SYSTEM_PROMPT, build_messages
from app.tools import TOOLS, FaultInjector, ToolContext, execute_tool

log = logging.getLogger('agent')

CONSEQUENTIAL_FLAGS = {'purchase_or_payment', 'external_order_or_delivery', 'external_communication', 'account_change'}
INJECTED_INVALID_DECISION = '{"status": "teleport", "action": "delete_all_files", "arguments": "rm -rf /"}'
MAX_EVENTS = 80
MAX_RATE_LIMIT_WAIT = 10.0  # seconds; honour provider retry-after only while the run budget allows
_BOUNDARY_TEXT = {
    'purchase_or_payment': 'buy, check out or pay for anything',
    'external_order_or_delivery': 'place orders or arrange delivery with a store or service',
    'external_communication': 'send or share anything with other people or services',
    'account_change': 'access or change any account',
}


class DailyModelBudget:
    """Process-wide spend control: caps provider calls per calendar day."""
    def __init__(self):
        self.day, self.calls = date.today(), 0

    def try_consume(self) -> bool:
        if date.today() != self.day:
            self.day, self.calls = date.today(), 0
        if self.calls >= settings.daily_model_call_limit:
            return False
        self.calls += 1
        return True


daily_budget = DailyModelBudget()


def parse_decision(raw: str) -> tuple[AgentDecision | None, str | None, list[str]]:
    """Parse model text into an AgentDecision. Returns (decision, error, warnings)."""
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', (raw or '').strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find('{'), text.rfind('}')
        try:
            data = json.loads(text[start:end + 1]) if 0 <= start < end else None
        except json.JSONDecodeError:
            data = None
    if not isinstance(data, dict):
        return None, 'output is not a single valid JSON object', []
    unknown = sorted(set(data) - set(AgentDecision.model_fields))
    warnings = [f'ignored unknown top-level field(s): {", ".join(unknown)}'] if unknown else []
    data = {k: v for k, v in data.items() if k in AgentDecision.model_fields}
    if data.get('arguments') is None:
        data['arguments'] = {}
    try:
        return AgentDecision.model_validate(data), None, warnings
    except ValidationError as exc:
        details = '; '.join(f"{'.'.join(map(str, e['loc'])) or 'decision'}: {e['msg']}" for e in exc.errors()[:5])
        return None, details, warnings


def _observation_for_model(action: str, observation: dict) -> Any:
    """Compact tool observation for the next prompt (the full plan is rendered by the app)."""
    if action == 'build_grocery_plan':
        plan = observation['plan']
        return {'recipes': plan['recipes'], 'item_count': len(plan['items']),
                'excluded_items': plan['excluded_items'], 'notes': plan['notes']}
    if action == 'get_recipe':
        recipe = dict(observation['recipe'])
        recipe['notes_untrusted_text'] = recipe.pop('notes')
        return recipe
    return observation


class AgentRun:
    def __init__(self, request: ArenaRequest, history: list[BaseMessage], model: str,
                 sandbox: PlanSandbox | None, session_id: str | None, client: ModelClient | None):
        self.request, self.model, self.client = request, model, client
        self.history = history[-settings.max_history_messages:] if settings.max_history_messages else []
        self.started = time.monotonic()
        self.deadline = self.started + settings.run_timeout_seconds - 1.0  # leave room to respond
        self.faults = FaultInjector(request.arena_config.fault.type)
        start_sandbox = sandbox.model_copy(deep=True) if sandbox else PlanSandbox()
        goal = request.task if not start_sandbox.pending_goal else (
            f'{start_sandbox.pending_goal}\n[clarification answer] {request.task}')
        self.state = AgentState(request_id=request.request_id, session_id=session_id, user_goal=goal[:10000],
                                model=model, max_steps=request.arena_config.max_steps, sandbox=start_sandbox)
        user_turns = [str(m.content) for m in self.history if isinstance(m, HumanMessage)]
        self.grounding = GroundingText('\n'.join([*user_turns, start_sandbox.pending_goal or '', request.task]))
        self.current_request = GroundingText(request.task)
        self.ctx = ToolContext(catalog=load_catalog(), external=request.external_context)
        self.tool_calls: list[ToolTrace] = []
        self.events: list[dict] = []
        self.plan_built_this_run = False
        self.tokens_used = 0  # provider-reported where available, otherwise estimated (for budgeting only)

    # -- observability ---------------------------------------------------------------
    def emit(self, event: str, **fields: Any) -> None:
        record = {'step': self.state.current_step, 'event': event, **fields}
        if len(self.events) < MAX_EVENTS:
            self.events.append(record)
        log.info(json.dumps({'request_id': self.state.request_id, 'session_id': self.state.session_id, **record},
                            default=str)[:2000])

    def error(self, kind: str, detail: str) -> None:
        self.state.errors.append({'step': self.state.current_step, 'type': kind, 'detail': detail[:500]})

    # -- main loop -------------------------------------------------------------------
    async def run(self) -> ArenaResponse:
        s = self.state
        self.emit('run_start', goal=s.user_goal[:300], model=self.model, max_steps=s.max_steps,
                  fault=self.faults.fault, external_items=len(self.request.external_context),
                  resumed_clarification=bool(s.sandbox.pending_goal))
        for flag in detect_boundary_requests(self.request.task + '\n' + (s.sandbox.pending_goal or '')):
            self._flag(flag, 'deterministic_detector')
        if self.client is None:
            try:
                self.client = get_client(self.model)
            except LLMError as exc:
                return self.stop('failed', 'model_unavailable', 'No language model is configured for this '
                                 'deployment, so the planner cannot run. Nothing was changed.', error=('model_error', exc.detail))
        while True:
            if s.current_step >= s.max_steps:
                return self.stop('budget_exceeded', 'max_steps_reached', self._partial(
                    f'I reached the step limit ({s.max_steps}) before finishing.'))
            remaining = self.deadline - time.monotonic()
            if remaining < settings.min_model_window_seconds:
                return self.stop('budget_exceeded', 'time_budget_reached', self._partial('I ran out of time for this request.'))
            if self.tokens_used >= settings.max_run_tokens:
                return self.stop('budget_exceeded', 'token_budget_reached', self._partial('I reached the token budget for this request.'))
            s.current_step += 1
            raw = await self._decide(remaining)
            if isinstance(raw, ArenaResponse):
                return raw
            decision, problem, warnings = parse_decision(raw)
            for w in warnings:
                s.warnings.append(w)
            if decision is None:
                result = self._reject('schema_validation', problem, raw)
                if result:
                    return result
                continue
            self.emit('model_decision', status=decision.status, action=decision.action, arguments=decision.arguments,
                      reason=decision.reason[:200], boundary_flag=decision.boundary_flag)
            s.last_decision = decision.model_dump()
            if decision.boundary_flag:
                self._flag(decision.boundary_flag, 'model')
            problems = validate_semantics(decision, s, self.ctx.recipes(s.sandbox), len(self.request.external_context),
                                          self.grounding, self.current_request)
            if problems:
                if decision.action:
                    self.tool_calls.append(ToolTrace(step=s.current_step, tool=decision.action, outcome='rejected'))
                result = self._reject('semantic_validation', '; '.join(problems), None, decision)
                if result:
                    return result
                continue
            self.emit('validation', result='accepted', action=decision.action)
            s.repair_feedback = None
            if decision.action:
                result = await self._execute(decision)
                if result:
                    return result
                if decision.status != 'completed':
                    continue
            return self._finish(decision)

    async def _decide(self, remaining: float) -> str | ArenaResponse:
        """One model decision (or an injected invalid one). Returns raw text or a terminal response."""
        s = self.state
        if self.faults.take('decision'):
            self.emit('fault_injected', fault='invalid_agent_decision')
            return INJECTED_INVALID_DECISION
        messages, truncated = build_messages(
            s, self.request.task, self.history, self.ctx.recipes(s.sandbox), self.request.external_context,
            max_external_chars=settings.max_external_chars, steps_remaining=s.max_steps - s.current_step,
            repairs_remaining=settings.max_decision_repairs - s.repairs_used, seconds_remaining=remaining)
        if truncated and 'external content truncated' not in s.warnings:
            s.warnings.append('external content truncated')
        estimate = sum(len(str(m.content)) for m in messages) // 4
        if self.tokens_used + estimate + settings.max_output_tokens > settings.max_run_tokens:
            return self.stop('budget_exceeded', 'token_budget_reached', self._partial(
                'The next step would exceed the token budget for this request.'))
        attempts = 0
        while True:
            attempts += 1
            if not daily_budget.try_consume():
                return self.stop('budget_exceeded', 'daily_model_budget_reached', 'The daily model-call budget for '
                                 'this deployment is used up. Please try again tomorrow.')
            s.model_calls += 1
            timeout = max(1.0, min(settings.model_timeout_seconds, self.deadline - time.monotonic() - 0.5))
            try:
                result = await self.client.complete(messages, settings.max_output_tokens, timeout)
                break
            except LLMError as exc:
                self.emit('model_error', kind=exc.kind, attempt=attempts)
                self.error('model_error', f'{exc.kind}: {exc.detail}')
                retryable = exc.kind in ('timeout', 'rate_limited', 'provider_error')
                wait = min(exc.retry_after or 1.0, MAX_RATE_LIMIT_WAIT) if exc.kind == 'rate_limited' else 0.2
                if not retryable or attempts >= 2 or s.model_calls >= 6 or \
                        self.deadline - time.monotonic() - wait < settings.min_model_window_seconds + 1:
                    return self.stop('failed', f'model_{exc.kind}', self._partial(
                        'The language model is unavailable right now, so I stopped safely.'))
                await asyncio.sleep(wait)
        if result.input_tokens is not None:
            s.input_tokens = (s.input_tokens or 0) + result.input_tokens
        if result.output_tokens is not None:
            s.output_tokens = (s.output_tokens or 0) + result.output_tokens
        self.tokens_used += (result.input_tokens or estimate) + (result.output_tokens or len(result.text) // 4)
        served_by = result.model or self.model
        cost = estimate_cost(served_by, result.input_tokens, result.output_tokens)
        if cost is not None:
            s.estimated_cost_usd = round((s.estimated_cost_usd or 0) + cost, 8)
        self.emit('model_call', model=served_by, latency_ms=round(result.latency_ms, 1),
                  input_tokens=result.input_tokens, output_tokens=result.output_tokens)
        return result.text

    def _reject(self, kind: str, problem: str, raw: str | None, decision: AgentDecision | None = None) -> ArenaResponse | None:
        """Record an invalid decision; schedule a bounded repair or stop with a typed contract error."""
        s = self.state
        self.error(kind, problem)
        self.emit('validation', result='rejected', kind=kind, detail=problem[:400],
                  raw_excerpt=(raw or '')[:200] if raw is not None else None)
        s.step_log.append(StepRecord(step=s.current_step, status=decision.status if decision else 'invalid',
                                     action=decision.action if decision else None,
                                     arguments=decision.arguments if decision else {}, outcome='rejected',
                                     observation=problem[:600]))
        if s.repairs_used >= settings.max_decision_repairs:
            return self.stop('contract_error', 'contract_retries_exhausted', self._partial(
                'I could not produce a valid plan decision within the repair limit, so I stopped without guessing.'))
        s.repairs_used += 1
        s.repair_feedback = (f'[{kind}] {problem}'[:1000] + '\nDo not repeat a rejected action. If the request cannot be '
                             'done as asked (missing, unknown, too many, or not in the plan), use status '
                             '"needs_clarification" and explain briefly what is possible.')
        self.emit('repair_requested', attempt=s.repairs_used, limit=settings.max_decision_repairs)
        return None

    async def _execute(self, decision: AgentDecision) -> ArenaResponse | None:
        s = self.state
        args = decision.typed_arguments()
        self.emit('tool_call', tool=decision.action, arguments=args.model_dump())
        outcome = await execute_tool(decision.action, args, s.sandbox, self.ctx, step=s.current_step,
                                     faults=self.faults, max_retries=settings.max_tool_retries,
                                     timeout=settings.tool_timeout_seconds)
        self.tool_calls.extend(outcome.traces)
        for event in outcome.events:
            if len(self.events) < MAX_EVENTS:
                self.events.append(event)
            log.info(json.dumps({'request_id': s.request_id, **event}, default=str))
        if not outcome.ok:
            if outcome.traces and outcome.traces[-1].outcome == 'rejected':
                # A business rule refused it: let the model react (counts as a repair, bounded).
                return self._reject('tool_rejected', outcome.error or 'tool rejected the request', None, decision)
            s.step_log.append(StepRecord(step=s.current_step, status=decision.status, action=decision.action,
                                         arguments=decision.arguments, outcome='failed', observation=outcome.error))
            self.error('tool_failure', outcome.error or 'tool failed')
            return self.stop('tool_error', 'tool_failure', self._partial(
                f'The {decision.action} tool kept failing ({outcome.error}), so I stopped without changing your plan.'))
        s.sandbox = outcome.sandbox
        s.tools_used.append(decision.action)
        if TOOLS[decision.action].mutates_selection:
            s.selection_changed = True
        if decision.action == 'build_grocery_plan':
            self.plan_built_this_run = True
        observation = _observation_for_model(decision.action, outcome.observation)
        s.last_tool_result = observation
        s.step_log.append(StepRecord(step=s.current_step, status=decision.status, action=decision.action,
                                     arguments=decision.arguments, outcome='success', observation=observation))
        self.emit('observation', tool=decision.action, summary=json.dumps(observation, default=str)[:600])
        return None

    def _flag(self, flag: str, source: str) -> None:
        if flag not in self.state.boundary_flags:
            self.state.boundary_flags.append(flag)
            self.emit('boundary_detected', flag=flag, source=source)

    # -- termination -------------------------------------------------------------------
    def _safe_message(self, text: str | None) -> str:
        text = (text or '').strip()
        if text and leaks_system_prompt(text, SYSTEM_PROMPT, CANARY):
            self.state.warnings.append('response withheld: it reproduced privileged instructions')
            self.emit('output_guard', action='redacted')
            return 'I can’t share my internal instructions.'
        return text

    def _partial(self, prefix: str) -> str:
        sandbox = self.state.sandbox
        if sandbox.plan is not None and not sandbox.plan_stale and self.plan_built_this_run:
            return prefix + ' The plan built so far:\n' + render_plan(sandbox.plan, 1500)
        if sandbox.selected:
            names = ', '.join(f'{rid} ({n} servings)' for rid, n in sandbox.selected.items())
            return prefix + f' Recipes planned so far: {names}. The grocery list was not finalized.'
        return prefix

    def _finish(self, decision: AgentDecision) -> ArenaResponse:
        s = self.state
        message = self._safe_message(decision.user_message)
        consequential = [f for f in s.boundary_flags if f in CONSEQUENTIAL_FLAGS]
        if consequential:
            refusal = ('I can’t ' + ' or '.join(_BOUNDARY_TEXT[f] for f in consequential) +
                       ': this planner only prepares sandbox grocery plans, so nothing was purchased, ordered, paid or sent.')
            if self.plan_built_this_run and s.sandbox.plan:
                refusal += ' Here is a draft plan you can use yourself:\n' + render_plan(s.sandbox.plan, 1500)
            elif decision.status == 'needs_clarification' and message:
                refusal += f' If you want a draft plan instead: {message}'
            return self.stop('blocked', 'autonomy_boundary', refusal)
        if decision.status == 'needs_clarification':
            s.sandbox.pending_question = message[:800]
            s.sandbox.pending_goal = s.user_goal[:2000]
            return self.stop('needs_clarification', 'clarification_required', message, keep_pending=True)
        if decision.status == 'blocked':
            reason = 'out_of_scope' if 'out_of_domain' in s.boundary_flags else 'policy_blocked'
            return self.stop('blocked', reason, message or 'I can’t help with that request.')
        if decision.status == 'failed':
            return self.stop('failed', 'agent_reported_failure', self._partial(message or 'I could not complete this request.'))
        # completed: semantic validation already checked the completion condition
        parts = [message] if message else []
        if s.sandbox.plan is not None and not s.sandbox.plan_stale and (self.plan_built_this_run or not message):
            parts.append(render_plan(s.sandbox.plan))
        return self.stop('completed', 'goal_completed', '\n\n'.join(parts) or 'Done.')

    def stop(self, status: str, reason: str, message: str, *, error: tuple[str, str] | None = None,
             keep_pending: bool = False) -> ArenaResponse:
        s = self.state
        if error:
            self.error(*error)
        if not keep_pending and status in ('completed', 'blocked'):
            s.sandbox.pending_goal = s.sandbox.pending_question = None
        s.current_status, s.stop_reason = status, reason
        self.emit('agent_stop', status=status, reason=reason, steps=s.current_step, model_calls=s.model_calls,
                  input_tokens=s.input_tokens, output_tokens=s.output_tokens,
                  duration_ms=round((time.monotonic() - self.started) * 1000, 1))
        if s.warnings:
            self.events.append({'step': s.current_step, 'event': 'warnings', 'items': s.warnings[:10]})
        return ArenaResponse(
            request_id=s.request_id, status=status, final_response=(message or reason)[:2000], steps=s.current_step,
            stop_reason=reason, tool_calls=self.tool_calls[:30], errors=s.errors[:20], events=self.events,
            metrics=Metrics(model_calls=min(s.model_calls, 6), input_tokens=s.input_tokens,
                            output_tokens=s.output_tokens, estimated_cost_usd=s.estimated_cost_usd))


async def run_agent(request, history, model, *, sandbox: PlanSandbox | None = None, session_id: str | None = None,
                    client: ModelClient | None = None) -> ArenaResponse:
    """Run one bounded agent episode. When a chat `sandbox` is supplied it is updated in place
    with the final execution state so the next turn can continue the same plan."""
    run = AgentRun(request, history, model, sandbox, session_id, client)
    response = await run.run()
    if sandbox is not None:
        for name in PlanSandbox.model_fields:
            setattr(sandbox, name, getattr(run.state.sandbox, name))
    return response
