"""get_order_status -- a read tool. It executes directly and mutates nothing."""

import json
import re
from datetime import date, datetime
from functools import lru_cache
from typing import Any, Optional

from app.config import get_settings, today

# Statuses actually present in orders.json. Each is handled explicitly below;
# an unrecognised one is surfaced rather than guessed at.
KNOWN_STATUSES = {
    "in_transit",
    "delivered",
    "partially_shipped",
    "delayed",
    "lost_in_transit",
    "cancelled",
}

# Per policy 1.6 a lost parcel is a claim, not a return, and is handled by a
# human. The tool flags it so routing does not depend on the model noticing.
HUMAN_ONLY_STATUSES = {"lost_in_transit"}


def _strip_private(obj: Any) -> Any:
    """Drop keys like `_note_for_designers`.

    Those notes state the expected answer ("Return must be refused"). Feeding
    them to the model would leak the grader's key into the prompt and hand an
    attacker a text channel inside otherwise-trusted data.
    """
    if isinstance(obj, dict):
        return {k: _strip_private(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, list):
        return [_strip_private(v) for v in obj]
    return obj


@lru_cache
def _dataset() -> dict:
    raw = json.loads(get_settings().orders_path.read_text(encoding="utf-8"))
    return _strip_private(raw)


@lru_cache
def _orders_by_id() -> dict[str, dict]:
    return {o["order_id"]: o for o in _dataset()["orders"]}


@lru_cache
def _customers_by_id() -> dict[str, dict]:
    return {c["customer_id"]: c for c in _dataset()["customers"]}


def normalise_order_id(raw: str) -> str:
    """Customers write `4521`, `tr-4521`, `#TR 4521`; all mean TR-4521.

    This is string cleanup on an argument the model already extracted, not
    intent detection.
    """
    cleaned = re.sub(r"[^A-Za-z0-9]", "", str(raw)).upper()
    if cleaned.startswith("TR"):
        cleaned = cleaned[2:]
    return f"TR-{cleaned}" if cleaned else str(raw)


def _as_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).date()


def _days_between(start: Optional[date], end: date) -> Optional[int]:
    return None if start is None else (end - start).days


def _overdue_by(
    expected_on: Optional[date], delivered_on: Optional[date], now: date
) -> Optional[int]:
    if expected_on is None or delivered_on is not None:
        return None
    overdue = (now - expected_on).days
    return overdue if overdue > 0 else None


def get_order_status(order_id: str) -> dict:
    """Look up one order. Returns an error dict on a miss rather than raising."""
    normalised = normalise_order_id(order_id)
    order = _orders_by_id().get(normalised)
    if order is None:
        return {"error": "order not found", "order_id": order_id}

    now = today()
    delivered_on = _as_date(order.get("delivered_at"))
    expected_on = _as_date(order.get("expected_delivery"))
    status = order["status"]

    shipped_items = [i for i in order["items"] if i.get("shipped") is True]
    pending_items = [i for i in order["items"] if i.get("shipped") is False]

    result = {
        "order_id": order["order_id"],
        "customer_id": order["customer_id"],
        "status": status,
        "status_known": status in KNOWN_STATUSES,
        "requires_human": status in HUMAN_ONLY_STATUSES,
        "placed_at": order.get("placed_at"),
        "expected_delivery": order.get("expected_delivery"),
        "delivered_at": order.get("delivered_at"),
        "carrier": order.get("carrier"),
        "tracking_number": order.get("tracking_number"),
        "payment_method": order.get("payment_method"),
        "shipping_city": order.get("shipping_city"),
        "items": order["items"],
        "total": order["total"],
        "as_of": now.isoformat(),
        "days_since_delivery": _days_between(delivered_on, now),
        # Only meaningful once the date has actually passed; a not-yet-due
        # order reports None rather than a negative number the model has to
        # interpret.
        "days_past_expected": _overdue_by(expected_on, delivered_on, now),
    }

    if status == "cancelled":
        result["cancelled_at"] = order.get("cancelled_at")
        result["refund_status"] = order.get("refund_status")
    if status == "partially_shipped":
        result["shipped_items"] = [i["name"] for i in shipped_items]
        result["pending_items"] = [
            {"name": i["name"], "backorder_eta": i.get("backorder_eta")}
            for i in pending_items
        ]
    return result


def customer_for_order(order_id: str) -> Optional[str]:
    """Owning customer_id, or None if the order does not exist."""
    order = _orders_by_id().get(normalise_order_id(order_id))
    return order["customer_id"] if order else None


def customer_name(customer_id: Optional[str]) -> Optional[str]:
    record = _customers_by_id().get(customer_id or "")
    return record["name"] if record else None
