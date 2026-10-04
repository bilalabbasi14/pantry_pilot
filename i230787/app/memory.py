"""In-memory chat storage (single worker; lost on restart).

Memory        - bounded LangChain message history per session (conversation layer).
SessionStore  - the per-session PlanSandbox (execution state: planned recipes, last plan,
                pending clarification). Kept separate from messages on purpose.
"""
from collections import OrderedDict
from langchain_core.messages import HumanMessage, AIMessage
from app.models import PlanSandbox

MAX_SESSIONS = 100
MAX_MESSAGES = 12          # six recent turns
MAX_HISTORY_CHARS = 24000  # character ceiling (not a token counter)


class Memory:
    def __init__(self):
        self.sessions = OrderedDict()

    def get(self, session):
        return list(self.sessions.get(session, []))

    def add(self, session, user, assistant):
        messages = self.get(session) + [HumanMessage(content=user), AIMessage(content=assistant)]
        while len(messages) > MAX_MESSAGES or sum(len(str(m.content)) for m in messages) > MAX_HISTORY_CHARS:
            messages = messages[2:]
        self.sessions[session] = messages
        self.sessions.move_to_end(session)
        while len(self.sessions) > MAX_SESSIONS:
            self.sessions.popitem(last=False)

    def clear(self, session):
        self.sessions.pop(session, None)


class SessionStore:
    def __init__(self):
        self.sessions: OrderedDict[str, PlanSandbox] = OrderedDict()

    def get(self, session: str) -> PlanSandbox:
        sandbox = self.sessions.get(session)
        return sandbox.model_copy(deep=True) if sandbox else PlanSandbox()

    def put(self, session: str, sandbox: PlanSandbox) -> None:
        self.sessions[session] = sandbox.model_copy(deep=True)
        self.sessions.move_to_end(session)
        while len(self.sessions) > MAX_SESSIONS:
            self.sessions.popitem(last=False)

    def clear(self, session: str) -> None:
        self.sessions.pop(session, None)
