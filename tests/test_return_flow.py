"""The whole "then act on it" path, end to end through /chat.

The unit tests in test_returns.py prove the tool refuses what it should. These
prove the loop actually reaches for it -- that an eligible item gets raised
rather than narrated, and that a missing size produces a question rather than
a half-staged record.

The intent gate is stubbed (as everywhere) and the agent model is scripted, so
these assert routing and wiring, not the model's judgement.
"""

import json
from types import SimpleNamespace

import pytest

from app import main
from app.router import agent_loop
from app.router.intent_gate import Routing
from tests.conftest import say
from tests.test_citation_guard import _message, _tool_call

KURTA = "TR-KRT-033"
SHIRT = "TR-SHR-009"


@pytest.fixture
def scripted_agent(monkeypatch):
    """Script the tier-3 model while leaving the real loop and tools in place."""

    def install(replies):
        calls = {"n": 0}

        def create(**_kwargs):
            index = min(calls["n"], len(replies) - 1)
            calls["n"] += 1
            return SimpleNamespace(choices=[SimpleNamespace(message=replies[index])])

        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        monkeypatch.setattr(agent_loop, "get_client", lambda: client)
        return calls

    return install


def eligibility(order_id):
    return Routing(intent="eligibility_check", order_id=order_id, confidence=0.95)


def test_an_eligible_refund_is_raised_not_handed_to_a_human(client, routes, scripted_agent):
    """The behaviour the brief asks for: act, then confirm."""
    routes["can I return the kurta from TR-4530?"] = eligibility("TR-4530")
    scripted_agent([
        _message(tool_calls=[_tool_call("check_return_eligibility", {"order_id": "TR-4530"})]),
        _message(tool_calls=[_tool_call("raise_return_request", {
            "order_id": "TR-4530", "item_sku": KURTA, "resolution": "refund"})]),
        _message(content="Done — I've raised the return for your Block-Print Kurta on "
                         "TR-4530. Reverse pickup is free on serviceable pincodes "
                         "(Return pickup -> 5.1 Pickup)."),
    ])

    body = say(client, "flow-1", "can I return the kurta from TR-4530?")

    assert body["escalated"] is False
    assert "raise_return_request" in body["tools_called"]
    assert "raised the return" in body["response"]
    # The old default -- deferring to a person for something already done.
    assert "human colleague" not in body["response"]


def test_an_exchange_asks_for_a_size_then_stages_it(client, routes, scripted_agent):
    """Two turns: the missing size becomes a question, not a partial record."""
    routes["I want to exchange the shirt from TR-4528"] = eligibility("TR-4528")
    routes["size L please"] = eligibility("TR-4528")
    scripted_agent([
        # turn 1 -- eligibility, then an exchange with no size, which is refused
        _message(tool_calls=[_tool_call("check_return_eligibility", {"order_id": "TR-4528"})]),
        _message(tool_calls=[_tool_call("raise_return_request", {
            "order_id": "TR-4528", "item_sku": SHIRT, "resolution": "exchange"})]),
        _message(content="Your Oxford Shirt is final sale, so it's a size exchange "
                         "rather than a refund (Returns -> 2.4 Final sale items). "
                         "Which size would you like instead?"),
        # turn 2 -- the size arrives and the exchange is staged
        _message(tool_calls=[_tool_call("raise_return_request", {
            "order_id": "TR-4528", "item_sku": SHIRT,
            "resolution": "exchange", "requested_size": "L"})]),
        _message(content="Booked — I've raised the size exchange for your Oxford "
                         "Shirt in L. Reverse pickup is free on serviceable "
                         "pincodes (Return pickup -> 5.1 Pickup)."),
    ])

    first = say(client, "flow-2", "I want to exchange the shirt from TR-4528")
    assert first["escalated"] is False
    assert "which size" in first["response"].lower()
    assert "raised" not in first["response"].lower()
    # The refusal records what is outstanding, which is what tells the router
    # next turn that a bare "size L please" is an answer, not a new request.
    assert "size" in (main.store.get("flow-2").pending_question or "")

    second = say(client, "flow-2", "size L please")
    assert second["escalated"] is False
    assert "raise_return_request" in second["tools_called"]
    assert "exchange" in second["response"].lower()


def test_staging_a_return_satisfies_the_eligibility_guard(client, routes, scripted_agent):
    """raise_return_request re-derives eligibility, so it counts.

    Without this the guard would force a retry on every successful return, to
    re-derive a verdict the staging tool had already derived in Python.
    """
    routes["raise the return for TR-4530"] = eligibility("TR-4530")
    calls = scripted_agent([
        _message(tool_calls=[_tool_call("raise_return_request", {
            "order_id": "TR-4530", "item_sku": KURTA, "resolution": "refund"})]),
        _message(content="I've raised the return for your Block-Print Kurta on TR-4530."),
    ])

    body = say(client, "flow-3", "raise the return for TR-4530")

    assert body["escalated"] is False
    assert "check_return_eligibility" not in body["tools_called"]
    assert calls["n"] == 2  # no correction round was needed


def test_an_ineligible_item_is_explained_and_nothing_is_staged(client, routes, scripted_agent):
    """TR-4527 is jewellery. The tool refuses even though the loop asked."""
    routes["return the earrings from TR-4527"] = eligibility("TR-4527")
    scripted_agent([
        _message(tool_calls=[_tool_call("check_return_eligibility", {"order_id": "TR-4527"})]),
        _message(tool_calls=[_tool_call("raise_return_request", {
            "order_id": "TR-4527", "item_sku": "TR-EAR-042", "resolution": "refund"})]),
        _message(content="I'm sorry — jewellery can't be returned, for hygiene and "
                         "safety reasons (Returns -> 2.3 Non-returnable categories)."),
    ])

    body = say(client, "flow-4", "return the earrings from TR-4527")

    assert body["escalated"] is False
    assert "2.3" in body["response"]
    # The tool was called and refused; the session records the attempt.
    state = main.store.get("flow-4")
    assert "return_raised" in state.checks_performed


def test_staging_the_exchange_clears_the_pending_question(client, routes, scripted_agent):
    """Once the size arrives and the exchange is staged, nothing is outstanding."""
    routes["exchange the shirt from TR-4528 in size L"] = eligibility("TR-4528")
    scripted_agent([
        _message(tool_calls=[_tool_call("raise_return_request", {
            "order_id": "TR-4528", "item_sku": SHIRT,
            "resolution": "exchange", "requested_size": "L"})]),
        _message(content="Booked — the size exchange for your Oxford Shirt in L is raised."),
    ])

    say(client, "flow-5", "exchange the shirt from TR-4528 in size L")

    assert main.store.get("flow-5").pending_question is None
