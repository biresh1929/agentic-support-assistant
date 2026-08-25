"""check_return_eligibility: deterministic, so these assert exact verdicts."""

import pytest

from app.tools.eligibility import check_return_eligibility

# TR-4530 was delivered 2026-07-26: the clean in-window case.
DELIVERED = "2026-07-26"


def test_happy_path_return_is_eligible_for_refund():
    result = check_return_eligibility("TR-4530")
    assert result["eligible"] is True
    assert result["items"][0]["resolution"] == "refund"


@pytest.mark.parametrize(
    "as_of_date, day, expected",
    [
        ("2026-08-24", 29, True),   # day 29 -- inside
        ("2026-08-25", 30, True),   # day 30 -- the last eligible day
        ("2026-08-26", 31, False),  # day 31 -- policy 2.1 admits no exception
    ],
)
def test_return_window_boundary(as_of, as_of_date, day, expected):
    as_of(as_of_date)
    result = check_return_eligibility("TR-4530")

    assert result["days_since_delivery"] == day
    assert result["eligible"] is expected


def test_expired_window_is_refused_on_date_grounds():
    result = check_return_eligibility("TR-4523")  # delivered 54 days ago
    assert result["eligible"] is False
    assert "past the 30-day window" in " ".join(result["reasons"])


def test_jewellery_is_refused_on_category_not_date():
    """TR-4527 is 6 days old -- comfortably in window. 2.3 still excludes it."""
    result = check_return_eligibility("TR-4527")

    assert result["eligible"] is False
    assert result["days_since_delivery"] == 6
    reasons = " ".join(result["reasons"])
    assert "jewellery" in reasons.lower()
    assert "window has closed" not in reasons


def test_final_sale_is_exchange_only():
    result = check_return_eligibility("TR-4528")
    item = result["items"][0]

    assert result["eligible"] is True
    assert item["resolution"] == "exchange_only"
    assert "no refund" in " ".join(item["reasons"]).lower()


def test_mixed_order_is_assessed_per_item():
    """TR-4522: cotton tee returnable, ankle socks excluded by 2.3."""
    result = check_return_eligibility("TR-4522")
    by_name = {i["name"]: i for i in result["items"]}

    assert by_name["Everyday Cotton Tee"]["resolution"] == "refund"
    assert by_name["Ankle Socks 3-pack"]["eligible"] is False
    # Top-level eligible means "at least one item qualifies".
    assert result["eligible"] is True


def test_cancelled_order_cannot_be_returned():
    result = check_return_eligibility("TR-4529")
    assert result["eligible"] is False
    assert "2.6" in " ".join(result["reasons"])


def test_lost_parcel_is_a_claim_not_a_return():
    result = check_return_eligibility("TR-4526")
    assert result["eligible"] is False
    assert result["requires_human"] is True
    assert "1.6" in " ".join(result["reasons"])


@pytest.mark.parametrize("order_id", ["TR-4521", "TR-4524", "TR-4525"])
def test_undelivered_orders_need_a_delivery_date(order_id):
    result = check_return_eligibility(order_id)
    assert result["eligible"] is False
    assert result["needs_info"] is not None


def test_unknown_order_reports_needs_info():
    result = check_return_eligibility("TR-0000")
    assert result["eligible"] is False
    assert result["needs_info"] == ["a valid order ID"]


def test_cash_on_delivery_refund_defers_bank_details_to_a_human():
    """Policy 3.3 -- the assistant must never collect bank details in chat.

    TR-4528 is cash-on-delivery and eligible, so the COD branch actually fires.
    """
    result = check_return_eligibility("TR-4528")

    assert result["eligible"] is True
    assert result["needs_info"] is not None
    needs = " ".join(result["needs_info"])
    assert "secure link" in needs and "3.3" in needs


def test_card_payment_does_not_ask_for_bank_details():
    result = check_return_eligibility("TR-4530")  # credit card
    assert result["needs_info"] is None
