"""raise_return_request -- a staged tool, and the second half of "then act on it".

Same philosophy as escalate_to_human: it returns a structured record and
changes nothing external. No warehouse is told, no refund moves, no pickup is
booked. What it produces is the record a real returns system would consume.

The part that matters is what it does *not* do: it does not take the caller's
word that an item is returnable. It calls check_return_eligibility itself and
re-derives the verdict from the order record before staging anything. A model
that has convinced itself a final-sale item deserves a refund, or that
jewellery is returnable, cannot stage one by saying so -- exactly as it cannot
offer a discount, because no code path exists that would compute one. The
guarantee is structural rather than a line in a prompt asking nicely.
"""

from typing import Optional

from app.retrieval.index import get_index
from app.tools.eligibility import check_return_eligibility

VALID_RESOLUTIONS = {"refund", "exchange"}

# Clauses describing what happens after a return is raised. Retrieved rather
# than hard-coded so the figures the agent quotes ("2-3 business days") land in
# the turn's evidence and survive the citation guard.
PICKUP_QUERY = "free reverse pickup serviceable pincodes schedule window"
REFUND_TIMELINE_QUERY = "refund timelines after inspection business days"


def _clause(query: str) -> Optional[dict]:
    hits = get_index().search(query, k=1)
    if not hits:
        return None
    return {"citation": hits[0].chunk.citation, "text": hits[0].chunk.text}


def _next_steps(resolution: str) -> list[dict]:
    """The clauses that answer "what happens now?", so the agent need not guess."""
    steps = [c for c in (_clause(PICKUP_QUERY),) if c]
    if resolution == "refund":
        timeline = _clause(REFUND_TIMELINE_QUERY)
        if timeline:
            steps.append(timeline)
    return steps


def _refuse(reason_code: str, reasons: list, **extra) -> dict:
    refusal = {
        "staged": False,
        "error": reason_code,
        "reasons": reasons,
        "needs_info": None,
    }
    refusal.update(extra)
    return refusal


def raise_return_request(
    order_id: str,
    item_sku: str,
    resolution: str,
    requested_size: Optional[str] = None,
) -> dict:
    """Stage a return or exchange for one item, after re-checking eligibility."""
    eligibility = check_return_eligibility(order_id)
    resolved_id = eligibility.get("order_id", order_id)
    items = eligibility.get("items") or []

    # No item list at all means the order was blocked before per-item
    # assessment ran -- not found, cancelled, lost, or not yet delivered.
    # check_return_eligibility already explained why; pass that through rather
    # than inventing a second explanation.
    if not items:
        return _refuse(
            "order_not_returnable",
            eligibility.get("reasons") or [f"No returnable items on {resolved_id}."],
            order_id=resolved_id,
            item_sku=item_sku,
            requires_human=eligibility.get("requires_human", False),
        )

    wanted = str(item_sku).strip().upper()
    item = next(
        (i for i in items if str(i.get("sku", "")).strip().upper() == wanted), None
    )
    if item is None:
        return _refuse(
            "item_not_in_order",
            [f"{resolved_id} has no item with SKU {item_sku}."],
            order_id=resolved_id,
            item_sku=item_sku,
            available_skus=[i.get("sku") for i in items],
        )

    # The guard. The item's own verdict decides this, not the caller's claim.
    if not item.get("eligible"):
        return _refuse(
            "item_not_eligible",
            list(item.get("reasons") or []),
            order_id=resolved_id,
            item_sku=item.get("sku"),
            item_name=item.get("name"),
        )

    if resolution not in VALID_RESOLUTIONS:
        return _refuse(
            "invalid_resolution",
            [f"{resolution!r} is not a resolution this tool can stage."],
            order_id=resolved_id,
            item_sku=item.get("sku"),
            valid_resolutions=sorted(VALID_RESOLUTIONS),
        )

    permitted = item.get("resolution")

    # Final sale: policy 2.4 allows a size exchange and nothing else. Asking
    # for a refund is refused whatever the reason given, and the correct
    # resolution is returned so the agent can course-correct in one turn
    # rather than guessing at what would have worked.
    if permitted == "exchange_only" and resolution == "refund":
        return _refuse(
            "wrong_resolution",
            list(item.get("reasons") or []),
            order_id=resolved_id,
            item_sku=item.get("sku"),
            item_name=item.get("name"),
            correct_resolution="exchange_only",
        )

    exchanging = resolution == "exchange" or permitted == "exchange_only"

    # An exchange with no size is incomplete, not invalid. needs_info is the
    # signal for the agent to ask the customer rather than stage a half-record
    # or invent a size.
    if exchanging and not (requested_size or "").strip():
        return _refuse(
            "missing_requested_size",
            ["An exchange needs the size the customer wants before it can be raised."],
            order_id=resolved_id,
            item_sku=item.get("sku"),
            item_name=item.get("name"),
            needs_info=["requested_size"],
        )

    staged_resolution = "exchange" if exchanging else "refund"
    record = {
        "staged": True,
        "action": "return_raised",
        "status": "staged",
        "order_id": resolved_id,
        "item_sku": item.get("sku"),
        "item_name": item.get("name"),
        "resolution": staged_resolution,
        "requested_size": requested_size if exchanging else None,
        "next_steps": _next_steps(staged_resolution),
        "policy_basis": eligibility.get("policy_basis", []),
    }

    # Policy 4.4 limits an item to one exchange, with a second requiring human
    # approval. orders.json carries no exchange history -- no count, no prior
    # returns array, nothing -- so there is no way to tell a first exchange
    # from a second. Rather than guess, the record says plainly that the check
    # was not performed, so a downstream system knows to run it itself.
    if staged_resolution == "exchange":
        record["repeat_exchange_checked"] = False
        record["repeat_exchange_note"] = (
            "Policy 4.4 allows one exchange per item; a second needs human "
            "approval. No exchange history exists in the order record, so this "
            "was not verified here."
        )
    if eligibility.get("needs_info"):
        record["needs_info"] = eligibility["needs_info"]
    return record
