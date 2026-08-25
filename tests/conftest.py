"""Shared fixtures.

The intent gate is a live model call, so tests stub it. That keeps assertions
about routing, state and templates deterministic and runnable offline; the
model's own classification accuracy is covered separately by the tests marked
`live`, which are skipped without an API key.
"""

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import main  # noqa: E402
from app.router.agent_loop import AgentResult  # noqa: E402
from app.router.intent_gate import Routing  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_store():
    """Every test starts with no sessions, so state cannot leak between them."""
    main.store = type(main.store)()
    yield


@pytest.fixture
def routes(monkeypatch):
    """Script the intent gate: routes[message] = Routing(...).

    Anything unscripted falls through to `ambiguous`, matching how the real
    gate fails closed.
    """
    table: dict[str, Routing] = {}

    def fake_classify(message, state):
        return table.get(message, Routing(intent="ambiguous", confidence=0.0))

    monkeypatch.setattr(main, "classify", fake_classify)
    return table


@pytest.fixture
def client():
    return TestClient(main.app)


def say(client, session_id: str, message: str) -> dict:
    """One turn of conversation against the real endpoint."""
    response = client.post("/chat", json={"session_id": session_id, "message": message})
    assert response.status_code == 200, response.text
    return response.json()


requires_live_llm = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="needs a working GROQ_API_KEY",
)


@pytest.fixture
def agent(monkeypatch):
    """Stub tier 3 so no test reaches the live API by accident.

    Returns a recorder: `agent.calls` is the list of messages the loop was
    handed, and `agent.result` is what it replies with.
    """

    class Recorder:
        def __init__(self):
            self.calls: list[str] = []
            self.result = AgentResult(text="(stub agent answer)", escalated=False)

        def __call__(self, message, state):
            self.calls.append(message)
            return self.result

    recorder = Recorder()
    monkeypatch.setattr(main.agent_loop, "run", recorder)
    return recorder


@pytest.fixture(autouse=True)
def no_live_calls(monkeypatch, request):
    """Fail loudly if a non-live test tries to build an LLM client."""
    if "live" in request.keywords:
        return

    def explode(*_args, **_kwargs):
        raise AssertionError(
            "a test reached the live API; stub `routes`/`agent` or mark it @pytest.mark.live"
        )

    monkeypatch.setattr("app.llm.get_client", explode)


@pytest.fixture
def as_of(monkeypatch):
    """Move the assistant's clock, so return-window boundaries are testable.

    orders.json must not be edited, and the fixtures are pinned to 2026-07-29,
    so day-29/30/31 cases are built by shifting the clock rather than the data.
    """
    import datetime

    def _set(iso: str):
        day = datetime.date.fromisoformat(iso)
        for module in ("app.tools.orders", "app.tools.eligibility"):
            monkeypatch.setattr(f"{module}.today", lambda: day)

    return _set
