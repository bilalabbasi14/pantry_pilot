import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api import router
from app.config import ROOT, settings
from app.memory import Memory, SessionStore

logging.basicConfig(level=settings.log_level.upper(), format='%(asctime)s %(levelname)s %(name)s %(message)s')
logging.getLogger('httpx').setLevel(logging.WARNING)  # request URLs are fine, but keep logs focused


@asynccontextmanager
async def lifespan(app):
    app.state.memory = Memory()
    app.state.sessions = SessionStore()
    app.state.busy = set()
    app.state.run_slots = asyncio.Semaphore(settings.max_concurrent_runs)
    yield

app = FastAPI(title='Pantry Pilot - Recipe & Grocery Planner Agent', lifespan=lifespan)
app.include_router(router)
app.mount('/static', StaticFiles(directory=ROOT / 'app/static'), name='static')
