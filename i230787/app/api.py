import json
import time
from collections import defaultdict, deque
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import FileResponse
from app.config import ROOT, settings
from app.models import ArenaRequest, ArenaResponse, ChatRequest
from app.arena import execute
from app.llm import available_models, default_model_id
router = APIRouter()


class RateLimiter:
    """Per-client sliding one-minute window (admission control for paid model calls)."""
    def __init__(self, per_minute: int):
        self.per_minute, self.hits = per_minute, defaultdict(deque)

    def allow(self, key: str) -> bool:
        now, window = time.monotonic(), self.hits[key]
        while window and now - window[0] > 60:
            window.popleft()
        if len(window) >= self.per_minute:
            return False
        window.append(now)
        if len(self.hits) > 10000:
            self.hits.clear()
        return True


arena_limiter = RateLimiter(settings.arena_rate_limit_per_minute)
chat_limiter = RateLimiter(settings.chat_rate_limit_per_minute)


def _admit(limiter: RateLimiter, request: Request) -> None:
    if not limiter.allow(request.client.host if request.client else 'unknown'):
        raise HTTPException(429, 'Too many requests; please wait a minute')


@router.get('/')
def index(): return FileResponse(ROOT / 'app/static/index.html')


@router.get('/health')
def health():
    return {'status': 'ok', 'implementation': 'grocery-planner', 'model_configured': bool(available_models())}


@router.get('/arena/manifest')
def manifest(): return json.loads((ROOT / 'arena_manifest.json').read_text(encoding='utf-8'))


@router.get('/models')
def models():
    """Only configured models whose provider key is present; never exposes keys."""
    return {'models': available_models(), 'default': default_model_id()}


@router.post('/arena/run', response_model=ArenaResponse)
async def arena_run(payload: ArenaRequest, request: Request):
    _admit(arena_limiter, request)
    async with request.app.state.run_slots:
        return await execute(payload, model=default_model_id())


@router.post('/chat', response_model=ArenaResponse)
async def chat(payload: ChatRequest, request: Request):
    model = default_model_id() if payload.model in ('', 'unconfigured') else payload.model
    if model != 'unconfigured' and model not in models()['models']:
        raise HTTPException(400, 'Model is not enabled')
    _admit(chat_limiter, request)
    busy = request.app.state.busy
    if payload.session_id in busy: raise HTTPException(409, 'This chat is already running')
    busy.add(payload.session_id)
    memory, sessions = request.app.state.memory, request.app.state.sessions
    try:
        sandbox = sessions.get(payload.session_id)
        async with request.app.state.run_slots:
            result = await execute(payload, memory.get(payload.session_id), model,
                                   sandbox=sandbox, session_id=payload.session_id)
        sessions.put(payload.session_id, sandbox)
        memory.add(payload.session_id, payload.task, result.final_response)
        return result
    finally:
        busy.discard(payload.session_id)


@router.delete('/chat/{session_id}')
def reset(session_id: str, request: Request):
    if session_id in request.app.state.busy: raise HTTPException(409, 'Chat is running')
    request.app.state.memory.clear(session_id)
    request.app.state.sessions.clear(session_id)
    return {'status': 'cleared'}
