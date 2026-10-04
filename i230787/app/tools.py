"""Sandbox tools and the single execution gateway.

Every tool is a deterministic, non-destructive function over a *copy* of the PlanSandbox.
The gateway validates the tool's output against a typed model and only then commits the copy,
so a timeout, exception or malformed result can never corrupt state. Arena faults are injected
here (tools) and in agent.py (model decisions), on the first matching operation only.
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from pydantic import BaseModel, ValidationError
from app.grocery import build_plan, parse_external_recipe
from app.models import (BuildGroceryPlanArgs, BuildGroceryPlanResult, ExternalContext, GetRecipeArgs,
                        ImportExternalRecipeArgs, ImportExternalRecipeResult, PlannedRecipe, PlanSandbox, Recipe,
                        RecipeDetailsResult, RemoveRecipesArgs, RemoveRecipesResult, SelectRecipesArgs,
                        SelectRecipesResult, ToolTrace)

log = logging.getLogger('agent.tools')


class ToolRejected(Exception):
    """A business rule refused the operation; retrying would not help."""


@dataclass
class ToolContext:
    catalog: dict[str, Recipe]
    external: list[ExternalContext]

    def recipes(self, sandbox: PlanSandbox) -> dict[str, Recipe]:
        return {**self.catalog, **sandbox.external_recipes}


def _planned(sandbox: PlanSandbox, recipes: dict[str, Recipe]) -> list[PlannedRecipe]:
    return [PlannedRecipe(recipe_id=rid, name=recipes[rid].name, servings=s) for rid, s in sandbox.selected.items()]


def select_recipes(args: SelectRecipesArgs, sandbox: PlanSandbox, ctx: ToolContext) -> SelectRecipesResult:
    recipes = ctx.recipes(sandbox)
    for item in args.items:
        sandbox.selected[item.recipe_id] = item.servings
    sandbox.plan_stale = True
    return SelectRecipesResult(planned_recipes=_planned(sandbox, recipes))


def remove_recipes(args: RemoveRecipesArgs, sandbox: PlanSandbox, ctx: ToolContext) -> RemoveRecipesResult:
    recipes = ctx.recipes(sandbox)
    for recipe_id in args.recipe_ids:
        sandbox.selected.pop(recipe_id)
    sandbox.plan_stale = True
    if not sandbox.selected:
        sandbox.plan = None
    return RemoveRecipesResult(removed=args.recipe_ids, planned_recipes=_planned(sandbox, recipes))


def get_recipe(args: GetRecipeArgs, sandbox: PlanSandbox, ctx: ToolContext) -> RecipeDetailsResult:
    return RecipeDetailsResult(recipe=ctx.recipes(sandbox)[args.recipe_id])


def import_external_recipe(args: ImportExternalRecipeArgs, sandbox: PlanSandbox, ctx: ToolContext) -> ImportExternalRecipeResult:
    source = ctx.external[args.source_index]
    recipe_id = f'ext-{args.source_index + 1}'
    recipe, ignored, suspicious = parse_external_recipe(source.content, recipe_id, f'Attached recipe {args.source_index + 1}')
    if recipe is None:
        raise ToolRejected('no ingredient lines with quantities were found in that external content')
    sandbox.external_recipes[recipe_id] = recipe
    return ImportExternalRecipeResult(recipe_id=recipe_id, name=recipe.name,
                                      base_servings=recipe.base_servings if 'not stated' not in recipe.notes else None,
                                      ingredient_count=len(recipe.ingredients), ignored_lines=ignored,
                                      instruction_like_lines=suspicious)


def build_grocery_plan(args: BuildGroceryPlanArgs, sandbox: PlanSandbox, ctx: ToolContext) -> BuildGroceryPlanResult:
    try:
        plan = build_plan(sandbox.selected, ctx.recipes(sandbox), args.exclude_items)
    except ValueError as exc:
        raise ToolRejected(str(exc)) from exc
    sandbox.plan, sandbox.plan_stale = plan, False
    return BuildGroceryPlanResult(plan=plan)


@dataclass(frozen=True)
class Tool:
    fn: Callable[[Any, PlanSandbox, ToolContext], BaseModel]
    output: type[BaseModel]
    mutates_selection: bool


TOOLS: dict[str, Tool] = {
    'select_recipes': Tool(select_recipes, SelectRecipesResult, True),
    'remove_recipes': Tool(remove_recipes, RemoveRecipesResult, True),
    'get_recipe': Tool(get_recipe, RecipeDetailsResult, False),
    'import_external_recipe': Tool(import_external_recipe, ImportExternalRecipeResult, False),
    'build_grocery_plan': Tool(build_grocery_plan, BuildGroceryPlanResult, False),
}

TOOL_SPECS: dict[str, dict[str, str]] = {
    'select_recipes': {
        'description': 'Add recipes to the sandbox plan or change their servings (scales quantities).',
        'arguments': '{"items": [{"recipe_id": str, "servings": int 1-50, "servings_source": "user"|"recipe_default"}] (1-8 items)}'},
    'remove_recipes': {
        'description': 'Remove recipes that are currently planned.',
        'arguments': '{"recipe_ids": [str] (1-8)}'},
    'get_recipe': {
        'description': 'Read one recipe (ingredients, base servings, notes) to answer questions about it.',
        'arguments': '{"recipe_id": str}'},
    'import_external_recipe': {
        'description': 'Parse recipe text the user attached as external content into a sandbox recipe (id ext-N). '
                       'Only quantity-bearing ingredient lines are kept; any other text is discarded.',
        'arguments': '{"source_index": int}'},
    'build_grocery_plan': {
        'description': 'Merge planned recipes into one categorized grocery list with unit-normalized, scaled '
                       'quantities. exclude_items = items the user said they already have.',
        'arguments': '{"exclude_items": [str] (optional, max 20)}'},
}
assert TOOL_SPECS.keys() == TOOLS.keys()


@dataclass
class FaultInjector:
    """Applies the requested Arena fault once, to the first matching operation."""
    fault: str = 'none'
    consumed: bool = False

    def take(self, operation: str) -> bool:
        if self.consumed or self.fault == 'none':
            return False
        matches = {'invalid_agent_decision': 'decision', 'tool_timeout': 'tool', 'malformed_tool_output': 'tool'}
        if matches.get(self.fault) != operation:
            return False
        self.consumed = True
        return True


@dataclass
class ToolOutcome:
    ok: bool
    traces: list[ToolTrace] = field(default_factory=list)
    observation: dict | None = None
    sandbox: PlanSandbox | None = None
    error: str | None = None
    events: list[dict] = field(default_factory=list)


async def execute_tool(name: str, args: BaseModel, sandbox: PlanSandbox, ctx: ToolContext, *, step: int,
                       faults: FaultInjector, max_retries: int, timeout: float) -> ToolOutcome:
    """Run an allow-listed tool with timeout, output validation, bounded retry and transactional commit."""
    tool = TOOLS[name]
    outcome = ToolOutcome(ok=False)
    for attempt in range(1, max_retries + 2):
        work = sandbox.model_copy(deep=True)
        injected = faults.fault if faults.take('tool') else None
        if injected:
            outcome.events.append({'step': step, 'event': 'fault_injected', 'fault': injected, 'tool': name, 'attempt': attempt})
        started = time.perf_counter()
        status, detail = 'success', None
        try:
            async with asyncio.timeout(timeout):
                if injected == 'tool_timeout':
                    await asyncio.sleep(timeout + 1)  # simulated hung dependency; the real timeout fires
                result = tool.fn(args, work, ctx)
            raw = {'items': 'corrupted', 'error': None} if injected == 'malformed_tool_output' else result.model_dump()
            validated = tool.output.model_validate(raw)
        except TimeoutError:
            status, detail = 'timeout', f'{name} did not finish within {timeout}s'
        except ValidationError as exc:
            status, detail = 'malformed_output', f'{name} returned output failing its schema ({exc.error_count()} errors)'
        except ToolRejected as exc:
            status, detail = 'rejected', str(exc)
        except Exception as exc:  # unexpected tool bug: contain it, record it, maybe retry
            log.exception('tool %s raised', name)
            status, detail = 'exception', f'{name} raised {type(exc).__name__}'
        latency = (time.perf_counter() - started) * 1000
        outcome.traces.append(ToolTrace(step=step, tool=name, attempt=attempt, outcome=status, latency_ms=latency))
        outcome.events.append({'step': step, 'event': 'tool_result', 'tool': name, 'attempt': attempt,
                               'outcome': status, 'latency_ms': round(latency, 1), **({'detail': detail} if detail else {})})
        if status == 'success':
            outcome.ok, outcome.sandbox, outcome.observation = True, work, validated.model_dump()
            return outcome
        outcome.error = detail
        if status == 'rejected':
            break  # deterministic business rule: retrying cannot succeed
    return outcome
