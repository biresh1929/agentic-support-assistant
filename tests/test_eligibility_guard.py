"""The eligibility-bypass guard, and the single bounded retry built on it.

check_return_eligibility exists so that date arithmetic, category exclusions
and final-sale rules are decided in Python rather than inferred. Its tool
description says to always call it -- and the conversation harness caught the
model ignoring that and reasoning to a verdict from get_order_status plus raw
policy text instead. The answers happened to be right, which is luck, and not
depending on luck is the entire reason the tool exists.

The detector is not a pattern match over the reply. The intent gate has
already separated eligibility_check from policy_question before the loop runs,
so that classification is the signal, and these tests pin both that it fires
when it should and that it stays off ordinary policy questions.

Reuses the scripted-LLM harness from test_citation_guard so both guards are
driven the same way.
"""

import json
from types import SimpleNamespace

from app.router import agent_loop
from app.state.conversation_state import ConversationState
from tests.test_citation_guard import _message, _tool_call, scripted_llm  # noqa: F401

WINDOW_SEARCH = _message(
    tool_calls=[_tool_call("search_policy", {"query": "return window"})]
)
ORDER_LOOKUP = _message(
    tool_calls=[_tool_call("get_order_status", {"order_id": "TR-4530"})]
)
ELIGIBILITY_CALL = _message(
    tool_calls=[_tool_call("check_return_eligibility", {"order_id": "TR-4530"})]
)

# Grounded against the retrieved window clause, so the citation guard passes
# and the eligibility check is the only thing left that can reject it.
VERDICT = _message(
    content="Yes, TR-4530 can be returned -- it was delivered inside the "
            "30 calendar day window (Returns -> 2.1 Return window)."
)


def _eligibility_state(session_id):
    """A session mid-way through an eligibility question, as main.py leaves it."""
    state = ConversationState(session_id=session_id)
    state.intent = "eligibility_check"
    return state


def test_eligibility_bypass_is_retried_and_recovers(scripted_llm):
    """Answer without the tool, get corrected, call it, and be accepted."""
    calls = scripted_llm([
        ORDER_LOOKUP,
        WINDOW_SEARCH,
        VERDICT,           # verdict with no check_return_eligibility -- rejected
        ELIGIBILITY_CALL,  # the correction lands
        VERDICT,           # same answer, now properly derived
    ])
    state = _eligibility_state("elig-1")

    result = agent_loop.run("can I return the kurta from TR-4530?", state)

    assert "check_return_eligibility" in result.tools_used
    assert result.eligibility_retried is True
    assert result.eligibility_bypass_failed is False
    assert result.escalated is False
    assert "30 calendar day" in result.text
    assert "eligibility_bypass_retry" in state.checks_performed


def test_eligibility_bypass_persists_and_escalates(scripted_llm):
    """One retry, then hand over -- and do not ship the unverified verdict."""
    calls = scripted_llm([
        ORDER_LOOKUP,
        WINDOW_SEARCH,
        VERDICT,   # first bypass
        VERDICT,   # ignores the correction and bypasses again
    ])
    state = _eligibility_state("elig-2")

    result = agent_loop.run("can I return the kurta from TR-4530?", state)

    assert result.eligibility_bypass_failed is True
    assert result.escalated is True
    assert state.escalation_reason == "unverified_claim"
    assert "check_return_eligibility" not in result.tools_used
    # The rejected answer must not leak through the escalation message.
    assert "Yes" not in result.text
    assert "30 calendar day" not in result.text
    assert "can be returned" not in result.text
    assert calls["n"] == 4  # no third attempt


def test_policy_question_intent_does_not_require_the_eligibility_tool(scripted_llm):
    """The scoping test: a general rule question must pass through untouched.

    "How long is the return window" is answerable from the policy alone and is
    classified policy_question, not eligibility_check. If this guard fired here
    it would escalate every ordinary policy answer in the product.
    """
    calls = scripted_llm([
        WINDOW_SEARCH,
        _message(content="You have 30 calendar days from delivery "
                         "(Returns -> 2.1 Return window)."),
    ])
    state = ConversationState(session_id="elig-3")
    state.intent = "policy_question"

    result = agent_loop.run("how long is the return window?", state)

    assert result.eligibility_retried is False
    assert result.eligibility_bypass_failed is False
    assert result.escalated is False
    assert "30 calendar days" in result.text
    assert "check_return_eligibility" not in result.tools_used
    assert calls["n"] == 2  # answered first time, no correction round


def test_eligibility_check_calling_the_tool_first_try_is_unaffected(scripted_llm):
    """The normal case pays nothing: no extra round trip, no escalation."""
    calls = scripted_llm([
        ELIGIBILITY_CALL,
        WINDOW_SEARCH,
        VERDICT,
    ])
    state = _eligibility_state("elig-4")

    result = agent_loop.run("can I return the kurta from TR-4530?", state)

    assert result.eligibility_retried is False
    assert result.eligibility_bypass_failed is False
    assert result.escalated is False
    assert "check_return_eligibility" in result.tools_used
    assert "eligibility_bypass_retry" not in state.checks_performed
    assert calls["n"] == 3


def test_the_requirement_is_per_turn_not_per_conversation(scripted_llm):
    """A later turn on the same session must call the tool again.

    Deliberate: the tool is deterministic and needs no model call, its reasons
    field answers a "why" follow-up directly, and remembering an earlier call
    would go stale as soon as a correction changed which order is in play.
    """
    state = _eligibility_state("elig-5")
    state.record_check("eligibility_check")  # as if an earlier turn had called it

    scripted_llm([WINDOW_SEARCH, VERDICT, VERDICT])
    result = agent_loop.run("why can't I get my money back?", state)

    assert result.eligibility_bypass_failed is True
    assert result.escalated is True


def test_bypass_escalation_reaches_the_wire_through_chat(client, routes, monkeypatch):
    """End to end: the escalation surfaces on /chat like any other."""
    from app.router.intent_gate import Routing

    routes["can I return TR-4530?"] = Routing(
        intent="eligibility_check", order_id="TR-4530", confidence=0.95
    )

    def fake_run(message, state):
        state.escalate("unverified_claim")
        state.record_check("escalation_staged")
        return agent_loop.AgentResult(
            text=agent_loop.ELIGIBILITY_BYPASS_MESSAGE,
            escalated=True,
            eligibility_bypass_failed=True,
        )

    from app import main
    monkeypatch.setattr(main.agent_loop, "run", fake_run)

    body = client.post(
        "/chat", json={"session_id": "wire-1", "message": "can I return TR-4530?"}
    ).json()

    assert body["escalated"] is True
    assert body["escalation_reason"] == "unverified_claim"
    assert "human colleague" in body["response"]


# --------------------------- the forced retry ---------------------------

def _recording_llm(monkeypatch, replies):
    """Like scripted_llm, but records the tool_choice sent on each call."""
    seen = []

    def create(**kwargs):
        seen.append(kwargs.get("tool_choice"))
        return SimpleNamespace(
            choices=[SimpleNamespace(message=replies[min(len(seen) - 1, len(replies) - 1)])]
        )

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(agent_loop, "get_client", lambda: client)
    return seen


def test_the_bypass_retry_forces_the_tool_instead_of_asking(monkeypatch):
    """Asking cost a full reasoning round; forcing is one deterministic call."""
    seen = _recording_llm(monkeypatch, [ORDER_LOOKUP, WINDOW_SEARCH, VERDICT,
                                        ELIGIBILITY_CALL, VERDICT])
    state = _eligibility_state("force-1")

    result = agent_loop.run("can I return the kurta from TR-4530?", state)

    assert result.eligibility_retried is True
    assert result.eligibility_bypass_failed is False
    # Calls 1-3 reason freely; call 4 -- the one after the bypass -- is forced.
    assert seen[:3] == ["auto", "auto", "auto"]
    assert seen[3] == agent_loop.FORCE_ELIGIBILITY_TOOL
    # And the forcing is one-shot: the loop reasons freely again afterwards.
    assert seen[4] == "auto"


def test_the_citation_guard_retry_still_reasons_freely(monkeypatch):
    """Scoping: that retry may need search_policy again, so it must not be forced."""
    seen = _recording_llm(monkeypatch, [
        WINDOW_SEARCH,
        _message(content="You have 45 calendar days to return it."),   # ungrounded
        _message(content="You have 30 calendar days from delivery (Returns -> 2.1)."),
    ])
    state = ConversationState(session_id="force-2")
    state.intent = "policy_question"

    result = agent_loop.run("how long do I have?", state)

    assert result.guard_retried is True
    assert all(choice == "auto" for choice in seen), seen
