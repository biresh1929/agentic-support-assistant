"""The citation guard, and the single bounded retry built on it."""

import json
from types import SimpleNamespace

import pytest

from app.guardrails.citation_guard import verify
from app.router import agent_loop
from app.state.conversation_state import ConversationState

WINDOW_CHUNK = {
    "citation": "Returns -> 2.1 Return window",
    "clause": "2.1 Return window",
    "text": (
        "2. Returns -> 2.1 Return window\n\n**2.1 Return window.** Items may be "
        "returned within 30 calendar days of the delivery date."
    ),
}


# --------------------------- the guard itself ---------------------------

@pytest.mark.parametrize(
    "answer, ok",
    [
        ("You have 30 calendar days from delivery (Returns -> 2.1).", True),
        ("You have 45 calendar days to return it.", False),
        ("Returns close after 30 days 【Returns → 2.1 Return window】", True),
        ("See policy 4.4 for the details.", False),
        ("I can offer you 20% off for the trouble.", False),
        ("I'm not able to offer 20% off.", True),
        ("You're owed a 250 store credit.", False),
        ("I can give you 500 off your next order.", False),
    ],
)
def test_verify_detects_ungrounded_claims(answer, ok):
    assert verify(answer, [WINDOW_CHUNK], []).ok is ok


def test_both_citation_formats_are_accepted():
    """Bracket and paren styles are cosmetic; the clause number is the claim."""
    bracketed = verify("Within 30 days 【Returns → 2.1 Return window】", [WINDOW_CHUNK], [])
    parenthesised = verify("Within 30 days (Returns -> 2.1 Return window)", [WINDOW_CHUNK], [])
    assert bracketed.ok and parenthesised.ok


def test_figures_grounded_in_tool_results_are_allowed():
    """'delivered 15 days ago' comes from the order record, not the policy."""
    evidence = [json.dumps({"days_since_delivery": 15})]
    assert verify("It arrived 15 days ago.", [WINDOW_CHUNK], evidence).ok is True
    assert verify("It arrived 99 days ago.", [WINDOW_CHUNK], evidence).ok is False


# --------------------------- the bounded retry ---------------------------

def _message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def _tool_call(name, arguments):
    return SimpleNamespace(
        id=f"call_{name}",
        function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
    )


@pytest.fixture
def scripted_llm(monkeypatch):
    """Drive the loop with a fixed sequence of model replies."""

    def install(replies):
        calls = {"n": 0}

        def create(**_kwargs):
            index = min(calls["n"], len(replies) - 1)
            calls["n"] += 1
            return SimpleNamespace(choices=[SimpleNamespace(message=replies[index])])

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        monkeypatch.setattr(agent_loop, "get_client", lambda: client)
        return calls

    return install


SEARCH = _message(tool_calls=[_tool_call("search_policy", {"query": "return window"})])


def test_ungrounded_answer_is_retried_once_and_can_recover(scripted_llm):
    calls = scripted_llm([
        SEARCH,
        _message(content="You have 45 calendar days to return it."),
        _message(content="You have 30 calendar days from delivery (Returns -> 2.1)."),
    ])
    state = ConversationState(session_id="guard-1")

    result = agent_loop.run("how long do I have to return this?", state)

    assert result.guard_retried is True
    assert result.guard_failed is False
    assert "30 calendar days" in result.text
    assert "citation_guard_retry" in state.checks_performed
    assert calls["n"] == 3  # search, bad answer, corrected answer


def test_second_failure_escalates_and_does_not_retry_again(scripted_llm):
    """The cap is the point: one retry, then hand over."""
    calls = scripted_llm([
        SEARCH,
        _message(content="You have 45 calendar days to return it."),
        _message(content="Actually it is 60 calendar days."),
    ])
    state = ConversationState(session_id="guard-2")

    result = agent_loop.run("how long do I have?", state)

    assert result.guard_failed is True
    assert result.escalated is True
    assert "45" not in result.text and "60" not in result.text
    assert state.escalation_reason == "unverified_claim"
    assert calls["n"] == 3  # never a fourth attempt


def test_ungrounded_answer_with_nothing_retrieved_escalates_immediately(scripted_llm):
    """With no retrieved text there is nothing to correct against, so no retry."""
    calls = scripted_llm([_message(content="You have 45 calendar days to return it.")])
    state = ConversationState(session_id="guard-3")

    result = agent_loop.run("how long do I have?", state)

    assert result.guard_retried is False
    assert result.escalated is True
    assert calls["n"] == 1


def test_grounded_answer_passes_through_untouched(scripted_llm):
    scripted_llm([
        SEARCH,
        _message(content="You have 30 calendar days from delivery (Returns -> 2.1)."),
    ])
    state = ConversationState(session_id="guard-4")

    result = agent_loop.run("how long do I have?", state)

    assert result.guard_retried is False
    assert result.guard_failed is False
    assert "30 calendar days" in result.text


def test_clauses_grounded_by_a_deterministic_tool_are_accepted():
    """check_return_eligibility resolves 2.3 in Python and reports it.

    Citing it is grounded even when search_policy was never called, so the
    guard must not treat retrieval as the only source of truth.
    """
    evidence = [
        json.dumps(
            {
                "order_id": "TR-4527",
                "eligible": False,
                "reasons": ["Pearl Drop Earrings is jewellery, which policy 2.3 makes non-returnable."],
                "policy_basis": ["Returns -> 2.3 Non-returnable categories"],
            }
        )
    ]
    answer = "Jewellery can't be returned for hygiene reasons (Returns → 2.3)."

    assert verify(answer, [], evidence).ok is True


def test_a_clause_in_no_tool_output_is_still_caught():
    evidence = [json.dumps({"order_id": "TR-4527", "eligible": False})]
    assert verify("See policy 4.4 about exchanges.", [], evidence).ok is False
