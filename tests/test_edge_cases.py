"""Two things that must not go wrong: bank details, and junk input.

Neither needs a live model. The COD tests assert what the tools deterministically
surface and that the refusal path is actually *available* to the agent -- if
policy 3.3 were not in the turn's evidence, the citation guard would reject a
refusal that cited it and escalate instead of refusing cleanly.
"""

from types import SimpleNamespace

import pytest

from app import main
from app.prompts.system import SYSTEM_PROMPT
from app.router import agent_loop
from app.router.intent_gate import Routing
from app.tools.eligibility import check_return_eligibility
from app.tools.returns import raise_return_request
from tests.conftest import say
from tests.test_citation_guard import _message, _tool_call

COD_ORDER = "TR-4528"      # C-103, cash on delivery, final sale
SHIRT = "TR-SHR-009"
ACCOUNT_NUMBER = "50100234567890"


@pytest.fixture
def scripted_agent(monkeypatch):
    def install(replies):
        calls = {"n": 0}

        def create(**_kwargs):
            index = min(calls["n"], len(replies) - 1)
            calls["n"] += 1
            return SimpleNamespace(choices=[SimpleNamespace(message=replies[index])])

        monkeypatch.setattr(
            agent_loop, "get_client",
            lambda: SimpleNamespace(
                chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
        )
        return calls

    return install


# --------------------------------------------------------------------------
# Cash-on-delivery refunds: bank details are a human's job, over a secure link
# --------------------------------------------------------------------------

def test_a_cod_order_surfaces_the_secure_link_route_not_a_request_for_details():
    """The tool names the process, so the agent never has to improvise one."""
    result = check_return_eligibility(COD_ORDER)

    assert result["needs_info"], "a COD refund must flag how bank details are collected"
    guidance = " ".join(result["needs_info"]).lower()
    assert "secure link" in guidance
    assert "human agent" in guidance
    assert "3.3" in guidance


def test_staging_a_cod_return_carries_the_same_guidance_forward():
    staged = raise_return_request(COD_ORDER, SHIRT, "exchange", requested_size="L")

    assert staged["staged"] is True
    assert "secure link" in " ".join(staged.get("needs_info") or []).lower()


def test_the_prompt_forbids_collecting_bank_details_in_chat():
    lowered = SYSTEM_PROMPT.lower()
    assert "never ask for or accept bank account numbers" in lowered
    assert "secure link" in lowered


def test_volunteered_bank_details_are_refused_and_not_echoed_back(
    client, routes, scripted_agent
):
    """The customer supplies an account number unprompted.

    What this pins is that the refusal is *reachable*: citing 3.3 survives the
    citation guard because check_return_eligibility put that clause in the
    turn's evidence. Without it the guard would treat the citation as
    ungrounded and escalate instead of answering.
    """
    message = f"just refund my COD order {COD_ORDER}, my account is {ACCOUNT_NUMBER}"
    routes[message] = Routing(
        intent="eligibility_check", order_id=COD_ORDER, confidence=0.95
    )
    scripted_agent([
        _message(tool_calls=[_tool_call("check_return_eligibility",
                                        {"order_id": COD_ORDER})]),
        _message(content="I can't take bank details over chat — a human colleague "
                         "collects those on a secure link (Refunds -> 3.3 "
                         "Cash-on-delivery refunds). Please don't share your "
                         "account number here."),
    ])

    body = say(client, "cod-1", message)

    assert body["escalated"] is False, "a clean refusal, not a handoff for grounding"
    assert ACCOUNT_NUMBER not in body["response"]
    assert "3.3" in body["response"]
    assert "secure link" in body["response"].lower()


def test_an_answer_citing_33_without_retrieving_it_is_still_rejected(scripted_agent):
    """The guard is not disabled here just because the topic is sensitive."""
    from app.state.conversation_state import ConversationState

    scripted_agent([
        _message(content="Send your account number and we'll refund it "
                         "(Refunds -> 3.3 Cash-on-delivery refunds)."),
    ])
    state = ConversationState(session_id="cod-2")
    state.intent = "policy_question"

    result = agent_loop.run("how do COD refunds work?", state)

    assert result.escalated is True
    assert result.guard_failed is True
    assert ACCOUNT_NUMBER not in result.text


# --------------------------------------------------------------------------
# Junk input: never a 500
# --------------------------------------------------------------------------

def test_an_empty_message_is_rejected_by_validation_not_a_crash(client):
    response = client.post("/chat", json={"session_id": "junk-1", "message": ""})
    assert response.status_code == 422  # schema violation, handled
    assert "detail" in response.json()


def test_a_missing_message_field_is_rejected_cleanly(client):
    response = client.post("/chat", json={"session_id": "junk-2"})
    assert response.status_code == 422


def test_an_over_long_message_is_rejected_cleanly(client):
    response = client.post(
        "/chat", json={"session_id": "junk-3", "message": "x" * 4001}
    )
    assert response.status_code == 422


@pytest.mark.parametrize(
    "message",
    [
        "asdkjhaskjdhkajshd",
        "   ",
        "?????",
        "🙂🙂🙂",
        "<script>alert(1)</script>",
        "'; DROP TABLE orders; --",
        "\\n\\t\\r",
    ],
    ids=["gibberish", "whitespace", "punctuation", "emoji", "html", "sqlish", "escapes"],
)
def test_junk_input_routes_to_ambiguous_and_asks_a_question(client, routes, message):
    """Unscripted messages fall through to ambiguous, matching the failing-closed
    gate. None of these may produce a 500 or an unhandled exception."""
    body = say(client, f"junk-{hash(message)}", message)   # say() asserts 200

    assert body["escalated"] is False       # first vague turn asks, never escalates
    assert body["response"].strip()
    assert "?" in body["response"]
    assert body["session_id"] == f"junk-{hash(message)}"


def test_two_junk_turns_escalate_rather_than_looping_forever(client, routes):
    first = say(client, "junk-loop", "asdkjhaskjdh")
    second = say(client, "junk-loop", "qwertyuiop")

    assert first["escalated"] is False
    assert second["escalated"] is True
    assert second["escalation_reason"] == "ambiguous_request"


def test_an_unknown_session_id_shape_is_still_accepted(client, routes):
    """session_id is opaque to the app; nothing may parse or trust its contents."""
    body = say(client, "../../etc/passwd", "asdkjhaskjdh")
    assert body["session_id"] == "../../etc/passwd"
    assert body["response"].strip()
