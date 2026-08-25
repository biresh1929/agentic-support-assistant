"""Tier 2: order-status lookups answered from a template, with no model call.

Roughly 70% of Trendly's volume is repetitive status checks. Once the router
has an intent and an order ID, the order record *is* the answer -- there is
nothing for a 70B model to reason about, so nothing calls one. Statuses whose
correct answer depends on policy (delayed, lost) are deliberately not served
here; they are handed to the agent loop or escalated instead.
"""

from datetime import date, datetime
from typing import Optional

FAST_PATH_STATUSES = {"in_transit", "delivered", "cancelled", "partially_shipped"}


def _pretty_date(value: Optional[str]) -> str:
    if not value:
        return "an unconfirmed date"
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return f"{parsed.day} {parsed:%B %Y}"


def _join(names: list[str]) -> str:
    if len(names) <= 1:
        return names[0] if names else ""
    return f"{', '.join(names[:-1])} and {names[-1]}"


def render_not_found(order_id: str) -> str:
    return (
        f"I couldn't find an order with the ID {order_id}. Trendly order numbers "
        "look like TR-4521 — could you double-check the number on your confirmation "
        "email? If it's right, I'll pass this to a human agent."
    )


def can_fast_path(order: dict) -> bool:
    """True when the status alone answers the question."""
    return "error" not in order and order.get("status") in FAST_PATH_STATUSES


def render(order: dict) -> str:
    """Templated natural language for one order record."""
    status = order["status"]
    oid = order["order_id"]
    carrier = order.get("carrier")
    tracking = order.get("tracking_number")

    if status == "in_transit":
        line = (
            f"Order {oid} is on its way. It's with {carrier} under tracking number "
            f"{tracking}, and is expected to arrive on "
            f"{_pretty_date(order.get('expected_delivery'))}."
        )
        return line + " Delivery estimates aren't guarantees, but this one is on schedule."

    if status == "delivered":
        delivered = _pretty_date(order.get("delivered_at"))
        days = order.get("days_since_delivery")
        line = f"Order {oid} was delivered on {delivered}"
        line += f" — {days} days ago." if days is not None else "."
        return line + " Let me know if you'd like to look at a return or exchange."

    if status == "cancelled":
        refund = order.get("refund_status")
        line = f"Order {oid} was cancelled on {_pretty_date(order.get('cancelled_at'))}."
        if refund == "processed":
            line += " The refund for it has already been processed."
        return line

    if status == "partially_shipped":
        shipped = _join(order.get("shipped_items", []))
        pending = order.get("pending_items", [])
        line = f"Order {oid} has shipped in two parts. "
        if shipped:
            line += (
                f"{shipped} is already on the way with {carrier} under tracking "
                f"{tracking}, expected {_pretty_date(order.get('expected_delivery'))}. "
            )
        if pending:
            names = _join([p["name"] for p in pending])
            eta = pending[0].get("backorder_eta")
            line += f"{names} is on backorder"
            line += f" and is expected to ship around {_pretty_date(eta)}." if eta else "."
            line += " There's no second shipping charge for the split delivery."
        return line.strip()

    raise ValueError(f"status {status!r} is not served by the fast path")
