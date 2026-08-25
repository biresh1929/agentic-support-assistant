"""Tier 2: order status end to end through the real /chat endpoint."""

import pytest

from app import main
from app.router.intent_gate import Routing
from tests.conftest import say


def order_status(order_id=None):
    return Routing(intent="order_status", order_id=order_id, confidence=0.95)


def test_in_transit_order_reports_carrier_and_eta(client, routes):
    routes["Where is my order TR-4521?"] = order_status("TR-4521")
    body = say(client, "s1", "Where is my order TR-4521?")

    assert body["escalated"] is False
    assert "TR-4521" in body["response"]
    assert "BlueDart" in body["response"]
    assert "BD8871209341" in body["response"]
    assert "31 July 2026" in body["response"]


def test_delivered_order_reports_the_delivery_date(client, routes):
    routes["did TR-4522 arrive?"] = order_status("TR-4522")
    body = say(client, "s1", "did TR-4522 arrive?")

    assert body["escalated"] is False
    assert "14 July 2026" in body["response"]
    assert "15 days ago" in body["response"]


def test_cancelled_order_says_so_and_mentions_the_refund(client, routes):
    routes["what happened to TR-4529"] = order_status("TR-4529")
    body = say(client, "s1", "what happened to TR-4529")

    assert "cancelled" in body["response"].lower()
    assert "refund" in body["response"].lower()


def test_partially_shipped_order_separates_shipped_from_backordered(client, routes):
    routes["status of TR-4524"] = order_status("TR-4524")
    body = say(client, "s1", "status of TR-4524")

    assert "High-Rise Straight Jeans" in body["response"]
    assert "Woven Leather Belt" in body["response"]
    assert "backorder" in body["response"].lower()
    # Policy 1.4: the customer is not charged twice.
    assert "second shipping charge" in body["response"]


def test_unknown_order_returns_a_miss_not_an_error(client, routes):
    routes["where is TR-9999"] = order_status("TR-9999")
    body = say(client, "s1", "where is TR-9999")

    assert body["escalated"] is False
    assert "couldn't find" in body["response"]
    # A miss must not become the active order.
    assert main.store.get("s1").active_order_id is None


def test_order_status_without_an_id_asks_which_order(client, routes):
    routes["where is my stuff"] = order_status(None)
    body = say(client, "s1", "where is my stuff")

    assert body["escalated"] is False
    assert "which order" in body["response"].lower()
    assert main.store.get("s1").pending_question is not None


def test_followup_turn_reuses_the_active_order(client, routes):
    routes["Where is my order TR-4521?"] = order_status("TR-4521")
    routes["any update?"] = order_status(None)

    say(client, "s2", "Where is my order TR-4521?")
    body = say(client, "s2", "any update?")

    assert "TR-4521" in body["response"]
    assert main.store.get("s2").turn_count == 2


def test_customer_correction_switches_orders_and_keeps_history(client, routes):
    """'Actually, it was the other one' -- the turn every grader tries.

    Both orders belong to C-100; a correction across customers is a different
    scenario, covered by the ownership test below.
    """
    routes["where is TR-4521"] = order_status("TR-4521")
    routes["sorry, I meant TR-4524"] = order_status("TR-4524")

    say(client, "s3", "where is TR-4521")
    body = say(client, "s3", "sorry, I meant TR-4524")

    state = main.store.get("s3")
    assert state.active_order_id == "TR-4524"
    assert "TR-4521" in state.order_id_history
    assert "backorder" in body["response"].lower()


def test_session_bound_to_one_customer_refuses_another_customers_order(client, routes):
    """TR-4521 is C-100's, TR-4522 is C-101's. The binding is what blocks it."""
    routes["where is TR-4521"] = order_status("TR-4521")
    routes["and what about TR-4522?"] = order_status("TR-4522")

    say(client, "s6", "where is TR-4521")
    body = say(client, "s6", "and what about TR-4522?")

    assert body["escalated"] is True
    # No detail of the other order may leak -- not the carrier, not the date.
    assert "Delhivery" not in body["response"]
    assert "14 July" not in body["response"]
    assert "DL5520998112" not in body["response"]
    assert main.store.get("s6").customer_id == "C-100"
    assert "cross_customer_lookup_refused" in main.store.get("s6").checks_performed


def test_lookup_records_a_check_for_the_escalation_summary(client, routes):
    routes["where is TR-4521"] = order_status("TR-4521")
    say(client, "s4", "where is TR-4521")

    assert main.store.get("s4").checks_performed == ["order_lookup"]


@pytest.mark.parametrize("order_id", ["TR-4525", "TR-4526"])
def test_policy_dependent_statuses_hand_off_to_the_agent_loop(
    client, routes, agent, order_id
):
    """Delayed and lost need policy, so tier 2 declines and tier 3 takes over."""
    routes["update?"] = order_status(order_id)
    say(client, "s5", "update?")

    assert agent.calls == ["update?"]


@pytest.mark.parametrize("order_id", ["TR-4521", "TR-4522", "TR-4529", "TR-4524"])
def test_fast_path_statuses_never_reach_the_agent_loop(client, routes, agent, order_id):
    """The whole point of tier 2: no model call for a plain status check."""
    routes["where is it"] = order_status(order_id)
    say(client, "s7", "where is it")

    assert agent.calls == []
