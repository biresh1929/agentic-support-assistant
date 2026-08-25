"""check_return_eligibility -- a deterministic function, not a model call.

Date arithmetic and category exclusions are exactly the things a language
model gets quietly wrong: 29 vs 31 days, "delivered" vs "dispatched", jewellery
being non-returnable even inside the window. All of it is computed here in
plain Python. The model's job is to call this and narrate the result, not to
decide it.

Eligibility is per item, because an order can be mixed: TR-4522 pairs a
returnable cotton tee with non-returnable socks, and a single boolean would
have to lie about one of them.
"""

from typing import Optional

from app.config import today
from app.guardrails.amounts import FOOTWEAR_BOX_DEDUCTION
from app.retrieval.index import get_index
from app.tools.orders import get_order_status

# policy 2.1
RETURN_WINDOW_DAYS = 30

# policy 2.3, mapped onto the category values used in orders.json.
NON_RETURNABLE_CATEGORIES = {
    "innerwear": "innerwear and socks",
    "jewellery": "jewellery",
    "beauty": "beauty and fragrance products",
    "fragrance": "beauty and fragrance products",
    "face_masks": "face masks",
    "gift_cards": "gift cards",
}

# Statuses from which no return can be raised at all, with the clause that says so.
BLOCKING_STATUSES = {
    "cancelled": (
        "Order TR was cancelled, and policy 2.6 allows no return against a "
        "cancelled order."
    ),
    "lost_in_transit": (
        "The carrier marked this parcel lost. Policy 1.6 makes that a "
        "lost-parcel claim handled by a human agent, not a return."
    ),
}

NOT_YET_DELIVERED = {"in_transit", "partially_shipped", "delayed"}

CITED_CLAUSES = (
    "return window 30 calendar days from delivery",
    "non-returnable categories hygiene",
    "final sale size exchange only",
    "footwear original shoe box",
)


def _policy_basis() -> list[str]:
    """Citations for the clauses this function encodes, pulled from the index."""
    citations: list[str] = []
    index = get_index()
    for query in CITED_CLAUSES:
        hits = index.search(query, k=1)
        if hits and hits[0].chunk.citation not in citations:
            citations.append(hits[0].chunk.citation)
    return citations


def _assess_item(item: dict, within_window: bool) -> dict:
    """Decide one line item. Returns its verdict and the reasons behind it."""
    category = str(item.get("category", "")).lower()
    reasons: list[str] = []

    excluded = NON_RETURNABLE_CATEGORIES.get(category)
    if excluded:
        return {
            "sku": item.get("sku"),
            "name": item.get("name"),
            "category": category,
            "eligible": False,
            "resolution": "none",
            "reasons": [
                f"{item.get('name')} is {excluded}, which policy 2.3 makes "
                "non-returnable for hygiene and safety reasons."
            ],
        }

    if not within_window:
        return {
            "sku": item.get("sku"),
            "name": item.get("name"),
            "category": category,
            "eligible": False,
            "resolution": "none",
            "reasons": ["The 30-day return window has closed (policy 2.1)."],
        }

    if item.get("final_sale"):
        return {
            "sku": item.get("sku"),
            "name": item.get("name"),
            "category": category,
            "eligible": True,
            "resolution": "exchange_only",
            "reasons": [
                f"{item.get('name')} is marked final sale. Policy 2.4 allows a "
                "size exchange only — no refund and no store credit."
            ],
        }

    if category == "footwear":
        reasons.append(
            "Footwear must go back in its original shoe box; policy 2.5 applies "
            f"a ₹{FOOTWEAR_BOX_DEDUCTION.rupees} deduction if the box is missing."
        )

    reasons.append("Within the 30-day window and a returnable category (policy 2.1).")
    return {
        "sku": item.get("sku"),
        "name": item.get("name"),
        "category": category,
        "eligible": True,
        "resolution": "refund",
        "reasons": reasons,
    }


def check_return_eligibility(order_id: str) -> dict:
    """Compute return eligibility for one order, item by item."""
    order = get_order_status(order_id)
    if "error" in order:
        return {
            "eligible": False,
            "reasons": [f"No order found with ID {order_id}."],
            "needs_info": ["a valid order ID"],
            "order_id": order_id,
        }

    status = order["status"]
    result: dict = {
        "order_id": order["order_id"],
        "eligible": False,
        "reasons": [],
        "needs_info": None,
        "items": [],
        "status": status,
        "as_of": today().isoformat(),
        "days_since_delivery": order.get("days_since_delivery"),
        "return_window_days": RETURN_WINDOW_DAYS,
        "policy_basis": _policy_basis(),
    }

    if status in BLOCKING_STATUSES:
        result["reasons"] = [BLOCKING_STATUSES[status].replace("TR", order["order_id"])]
        result["requires_human"] = status == "lost_in_transit"
        return result

    if status in NOT_YET_DELIVERED:
        result["reasons"] = [
            "The return window runs from the delivery date (policy 2.1), and "
            f"{order['order_id']} has not been delivered yet."
        ]
        result["needs_info"] = ["the delivery date, once the order arrives"]
        return result

    days = order.get("days_since_delivery")
    if days is None:
        result["reasons"] = ["This order is marked delivered but has no delivery date."]
        result["needs_info"] = ["the delivery date"]
        return result

    within_window = days <= RETURN_WINDOW_DAYS
    result["items"] = [_assess_item(item, within_window) for item in order["items"]]
    result["eligible"] = any(item["eligible"] for item in result["items"])

    if not within_window:
        result["reasons"].append(
            f"{order['order_id']} was delivered {days} days ago, past the "
            f"{RETURN_WINDOW_DAYS}-day window in policy 2.1. Requests after 30 "
            "days are not eligible under any circumstance."
        )
    else:
        result["reasons"].append(
            f"Delivered {days} days ago, inside the {RETURN_WINDOW_DAYS}-day window."
        )
    for item in result["items"]:
        result["reasons"].extend(item["reasons"])

    if result["eligible"] and order.get("payment_method") == "cash_on_delivery":
        # policy 3.3 -- a human collects bank details over a secure link.
        result["needs_info"] = [
            "bank details for a cash-on-delivery refund, which a human agent "
            "collects over a secure link (policy 3.3)"
        ]

    return result
