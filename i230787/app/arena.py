"""Common boundary; ordinary Arena requests have no chat memory and a fresh sandbox."""
import asyncio
import logging
from time import perf_counter
from app.agent import run_agent
from app.config import settings
from app.models import ArenaResponse
log = logging.getLogger('arena')


async def execute(request, history=None, model='unconfigured', *, sandbox=None, session_id=None, client=None):
    started = perf_counter()
    request = request.model_copy(deep=True)
    request.arena_config.max_steps = min(request.arena_config.max_steps, settings.max_steps)
    try:
        async with asyncio.timeout(settings.run_timeout_seconds):
            result = await run_agent(request, history or [], model, sandbox=sandbox, session_id=session_id, client=client)
            result = ArenaResponse.model_validate(result)
    except TimeoutError:
        result = ArenaResponse(request_id=request.request_id, status='budget_exceeded',
            final_response='The run timed out.', stop_reason='time_budget_reached')
    except Exception:
        log.exception('request=%s crashed', request.request_id)
        result = ArenaResponse(request_id=request.request_id, status='failed',
            final_response='The run stopped due to an internal error.', stop_reason='internal_error',
            errors=[{'type': 'internal_error', 'detail': 'unexpected exception; see server log'}])
    result.metrics.latency_ms = (perf_counter() - started) * 1000
    if len(result.model_dump_json().encode()) > 50000:
        result = ArenaResponse(request_id=request.request_id, status='failed',
            final_response='Response exceeded the size limit.', stop_reason='response_too_large')
    log.info('request=%s status=%s stop=%s steps=%s latency_ms=%.0f', request.request_id, result.status,
             result.stop_reason, result.steps, result.metrics.latency_ms)
    return result
