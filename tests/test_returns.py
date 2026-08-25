"""raise_return_request -- the staged action, and the guard inside it.

Every test here calls the tool directly, with arguments a model could have
invented. That is the point: the guarantee being tested is that the tool
re-derives eligibility itself rather than believing its caller, so it has to
hold when the caller is hostile, not merely when the caller is a well-behaved
agent loop.
"""

import pytest

from app.state.conversation_state import ConversationState
from app.tools.registry import dispatch
from app.tools.returns import raise_return_request

KURTA = "TR-KRT-033"      # TR-4530, C-101, delivered 3 days ago, plain refund
EARRINGS = "TR-EAR-042"   # TR-4527, C-102, jewellery -- never returnable
SHIRT = "TR-SHR-009"      # TR-4528, C-103, final sale -- exchange only


def test_raising_a_refund_for_an_eligible_item_succeeds():
    result = raise_return_request("TR-4530", KURTA, "refund")

    assert result["staged"] is True
    assert result["status"] == "staged"
    assert result["action"] == "return_raised"
    assert result["order_id"] == "TR-4530"
    assert result["item_sku"] == KURTA
    assert result["item_name"] == "Block-Print Kurta"
    assert result["resolution"] == "refund"
    assert result["requested_size"] is None
    # The record has to answer "what happens next?" without the model guessing.
    citations = " ".join(s["citation"] for s in result["next_steps"])
    assert "5.1" in citations          # free reverse pickup
    assert "3.1" in citations          # refund timeline


def test_raising_a_refund_for_an_ineligible_item_is_refused():
    """The guard, tested the only way that means anything: called directly.

    TR-4527 is jewellery. It is inside the 30-day window, so a model reasoning
    from the delivery date alone could talk itself into this. The tool asks
    check_return_eligibility and refuses regardless of what it was told.
    """
    result = raise_return_request("TR-4527", EARRINGS, "refund")

    assert result["staged"] is False
    assert result["error"] == "item_not_eligible"
    assert "status" not in result and "action" not in result
    # The refusal quotes the item's real reason, not a generic denial.
    joined = " ".join(result["reasons"]).lower()
    assert "hygiene" in joined
    assert "2.3" in joined


def test_exchange_without_a_size_asks_for_one_instead_of_staging():
    result = raise_return_request("TR-4528", SHIRT, "exchange")

    assert result["staged"] is False
    assert result["error"] == "missing_requested_size"
    assert result["needs_info"] == ["requested_size"]
    assert "status" not in result


def test_exchange_with_a_size_stages_correctly():
    result = raise_return_request("TR-4528", SHIRT, "exchange", requested_size="L")

    assert result["staged"] is True
    assert result["resolution"] == "exchange"
    assert result["requested_size"] == "L"
    assert result["item_name"] == "Oxford Shirt"
    # Policy 4.4 cannot be checked from this dataset; the record says so
    # rather than implying it was verified.
    assert result["repeat_exchange_checked"] is False
    assert "4.4" in result["repeat_exchange_note"]


def test_refund_requested_on_a_final_sale_item_is_refused_with_correction():
    result = raise_return_request("TR-4528", SHIRT, "refund")

    assert result["staged"] is False
    assert result["error"] == "wrong_resolution"
    assert result["correct_resolution"] == "exchange_only"
    assert "2.4" in " ".join(result["reasons"])


def test_a_size_exchange_on_an_ordinary_item_is_allowed():
    """Section 4.1 permits size exchanges generally, not only on final sale."""
    result = raise_return_request("TR-4530", KURTA, "exchange", requested_size="M")

    assert result["staged"] is True
    assert result["resolution"] == "exchange"
    assert result["requested_size"] == "M"


def test_an_unknown_sku_is_refused_and_lists_what_is_actually_there():
    result = raise_return_request("TR-4530", "TR-NOPE-999", "refund")

    assert result["staged"] is False
    assert result["error"] == "item_not_in_order"
    assert KURTA in result["available_skus"]


@pytest.mark.parametrize(
    "order_id, sku",
    [
        ("TR-4523", "TR-JKT-008"),   # delivered 54 days ago, outside the window
        ("TR-4529", "TR-SCF-027"),   # cancelled
        ("TR-4526", "TR-BAG-011"),   # lost in transit
        ("TR-9999", "TR-KRT-033"),   # no such order
    ],
)
def test_orders_that_cannot_produce_a_return_are_refused(order_id, sku):
    result = raise_return_request(order_id, sku, "refund")
    assert result["staged"] is False
    assert result["reasons"]


def test_an_invalid_resolution_is_refused():
    result = raise_return_request("TR-4530", KURTA, "store_credit")

    assert result["staged"] is False
    assert result["error"] == "invalid_resolution"
    assert result["valid_resolutions"] == ["exchange", "refund"]


# --------------------------------------------------------------------------
# The ownership check must apply here too, not merely be assumed to
# --------------------------------------------------------------------------

def test_cross_customer_order_id_is_refused_by_the_existing_ownership_check():
    """Asserted through dispatch, because that is where the guard lives.

    Adding a tool to ORDER_SCOPED_TOOLS is easy to believe you have done and
    easy to get wrong, so this exercises the real path rather than inspecting
    the set.
    """
    state = ConversationState(session_id="own-1")
    dispatch(state, "get_order_status", {"order_id": "TR-4530"})   # binds C-101
    assert state.customer_id == "C-101"

    result = dispatch(state, "raise_return_request", {
        "order_id": "TR-4523",       # C-102's bomber jacket
        "item_sku": "TR-JKT-008",
        "resolution": "refund",
    })

    assert result["error"] == "order does not belong to this customer"
    assert result.get("staged") is not True
    assert "Quilted Bomber Jacket" not in str(result)
    assert state.customer_id == "C-101"


def test_dispatch_records_the_check_and_tracks_the_order():
    state = ConversationState(session_id="own-2")
    result = dispatch(state, "raise_return_request", {
        "order_id": "TR-4530", "item_sku": KURTA, "resolution": "refund",
    })

    assert result["staged"] is True
    assert "return_raised" in state.checks_performed
    assert state.active_order_id == "TR-4530"


def test_the_tool_stages_nothing_outside_itself():
    """Staged means staged: calling it twice changes no shared state."""
    first = raise_return_request("TR-4530", KURTA, "refund")
    second = raise_return_request("TR-4530", KURTA, "refund")
    assert first == second
