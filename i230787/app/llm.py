"""Thin, dependency-free model clients (Gemini and Groq over httpx).
The model only ever returns text; app/agent.py parses and validates it before anything runs."""
import re
import time
from dataclasses import dataclass
from typing import Protocol
import httpx
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from app.config import settings

# USD per 1M tokens (input, output): paid-tier list prices looked up 2026-10-04 (sources in README).
# Used only for the estimated_cost_usd metric; free-tier calls are not actually billed.
PRICING_PER_MILLION = {
    'gemini:gemini-2.5-flash': (0.30, 2.50),
    'gemini:gemini-3.5-flash-lite': (0.30, 2.50),
    'groq:openai/gpt-oss-120b': (0.15, 0.75),
    'groq:openai/gpt-oss-20b': (0.10, 0.50),
}
KNOWN_PROVIDERS = ('gemini', 'groq')


@dataclass
class LLMResult:
    text: str
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: float
    model: str | None = None  # provider:model that actually answered (set by FallbackClient)


class LLMError(Exception):
    """kind: timeout | rate_limited | provider_error | unavailable"""
    def __init__(self, kind: str, detail: str, retry_after: float | None = None):
        super().__init__(f'{kind}: {detail}')
        self.kind, self.detail, self.retry_after = kind, detail, retry_after


class ModelClient(Protocol):
    async def complete(self, messages: list[BaseMessage], max_output_tokens: int, timeout: float) -> LLMResult: ...


def estimate_cost(model_id: str, input_tokens: int | None, output_tokens: int | None) -> float | None:
    price = PRICING_PER_MILLION.get(model_id)
    if price is None or input_tokens is None or output_tokens is None:
        return None
    return round((input_tokens * price[0] + output_tokens * price[1]) / 1_000_000, 8)


def _split_system(messages: list[BaseMessage]) -> tuple[str, list[BaseMessage]]:
    system = '\n\n'.join(str(m.content) for m in messages if isinstance(m, SystemMessage))
    return system, [m for m in messages if not isinstance(m, SystemMessage)]


async def _post(url: str, headers: dict, body: dict, timeout: float) -> tuple[dict, float]:
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(url, headers=headers, json=body)
    except httpx.TimeoutException as exc:
        raise LLMError('timeout', 'model call exceeded its time limit') from exc
    except httpx.HTTPError as exc:
        raise LLMError('provider_error', type(exc).__name__) from exc
    latency = (time.perf_counter() - started) * 1000
    if response.status_code == 429:
        hint = response.headers.get('retry-after') or ''.join(re.findall(r'"retryDelay":\s*"([\d.]+)s"', response.text)[:1])
        try:
            retry_after = float(hint)
        except ValueError:
            retry_after = None
        raise LLMError('rate_limited', 'provider rate limit or quota reached', retry_after)
    if response.status_code >= 400:
        # Provider error bodies never contain our key; keep only a short excerpt.
        raise LLMError('provider_error', f'HTTP {response.status_code}: {response.text[:200]}')
    try:
        return response.json(), latency
    except ValueError as exc:
        raise LLMError('provider_error', 'non-JSON provider response') from exc


class GeminiClient:
    def __init__(self, model: str, api_key: str):
        self.model, self.api_key = model, api_key

    async def complete(self, messages, max_output_tokens, timeout):
        system, rest = _split_system(messages)
        contents = []
        for message in rest:
            role = 'model' if isinstance(message, AIMessage) else 'user'
            if contents and contents[-1]['role'] == role:
                contents[-1]['parts'].append({'text': str(message.content)})
            else:
                contents.append({'role': role, 'parts': [{'text': str(message.content)}]})
        config = {'temperature': 0, 'maxOutputTokens': max_output_tokens, 'responseMimeType': 'application/json'}
        if self.model.startswith('gemini-2.5'):
            config['thinkingConfig'] = {'thinkingBudget': 0}  # decisions are short; spend no tokens on thinking
        elif self.model.startswith('gemini-3'):
            config['thinkingConfig'] = {'thinkingLevel': 'minimal'}
        body = {'systemInstruction': {'parts': [{'text': system}]}, 'contents': contents, 'generationConfig': config}
        url = f'https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent'
        data, latency = await _post(url, {'x-goog-api-key': self.api_key}, body, timeout)
        usage = data.get('usageMetadata') or {}
        try:
            text = ''.join(p.get('text', '') for p in data['candidates'][0]['content']['parts'])
        except (KeyError, IndexError, TypeError):
            text = ''  # e.g. safety block or MAX_TOKENS with no content; the validator rejects it
        output_tokens = usage.get('candidatesTokenCount')
        if output_tokens is not None:
            output_tokens += usage.get('thoughtsTokenCount', 0)
        return LLMResult(text, usage.get('promptTokenCount'), output_tokens, latency)


class GroqClient:
    def __init__(self, model: str, api_key: str):
        self.model, self.api_key = model, api_key

    async def complete(self, messages, max_output_tokens, timeout):
        role = {'system': 'system', 'human': 'user', 'ai': 'assistant'}
        body = {'model': self.model, 'temperature': 0, 'max_completion_tokens': max_output_tokens,
                'response_format': {'type': 'json_object'},
                'messages': [{'role': role.get(m.type, 'user'), 'content': str(m.content)} for m in messages]}
        if self.model.startswith('openai/gpt-oss'):
            body['reasoning_effort'] = 'low'
        data, latency = await _post('https://api.groq.com/openai/v1/chat/completions',
                                    {'Authorization': f'Bearer {self.api_key}'}, body, timeout)
        usage = data.get('usage') or {}
        try:
            text = data['choices'][0]['message'].get('content') or ''
        except (KeyError, IndexError, TypeError):
            text = ''
        return LLMResult(text, usage.get('prompt_tokens'), usage.get('completion_tokens'), latency)


def _api_key(provider: str) -> str:
    key = {'gemini': settings.gemini_api_key, 'groq': settings.groq_api_key}.get(provider)
    return key.get_secret_value().strip() if key else ''


def default_model_id() -> str:
    if settings.model_provider in KNOWN_PROVIDERS and settings.model_name:
        return f'{settings.model_provider}:{settings.model_name}'
    return 'unconfigured'


def available_models() -> list[str]:
    """Configured models whose provider key is present. Never includes key material."""
    candidates = [default_model_id()] + [m.strip() for m in settings.alternate_models.split(',') if m.strip()]
    result = []
    for model_id in candidates:
        provider, _, name = model_id.partition(':')
        if provider in KNOWN_PROVIDERS and name and _api_key(provider) and model_id not in result:
            result.append(model_id)
    return result


def _single_client(model_id: str) -> ModelClient:
    provider, _, name = model_id.partition(':')
    cls = GeminiClient if provider == 'gemini' else GroqClient
    return cls(name, _api_key(provider))


# Process-wide cooldowns after a provider says "rate limited" (circuit breaker).
_cooldown_until: dict[str, float] = {}


class FallbackClient:
    """Calls the primary model; on a rate limit or provider error it falls back to the next
    configured model (a different free-tier quota) instead of failing the whole run."""
    def __init__(self, model_ids: list[str]):
        self.model_ids = model_ids

    async def complete(self, messages, max_output_tokens, timeout):
        deadline = time.monotonic() + timeout
        last: LLMError | None = None
        candidates = [m for m in self.model_ids if _cooldown_until.get(m, 0) <= time.monotonic()] or self.model_ids[:1]
        for model_id in candidates:
            remaining = deadline - time.monotonic()
            if remaining < 1:
                break
            try:
                result = await _single_client(model_id).complete(messages, max_output_tokens, remaining)
                result.model = model_id
                return result
            except LLMError as exc:
                last = exc
                if exc.kind == 'rate_limited':
                    _cooldown_until[model_id] = time.monotonic() + min(exc.retry_after or 20, 60)
                if exc.kind == 'timeout':
                    break  # the time is spent; let the agent decide whether a retry fits its budget
        raise last or LLMError('timeout', 'no time left for a model call')


def get_client(model_id: str) -> ModelClient:
    """Client for the selected model, with the other configured models as fallbacks if enabled."""
    configured = available_models()
    if model_id not in configured:
        raise LLMError('unavailable', f'model {model_id!r} is not configured')
    if not settings.enable_model_fallback:
        return FallbackClient([model_id])
    return FallbackClient([model_id] + [m for m in configured if m != model_id])
