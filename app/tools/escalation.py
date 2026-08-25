"""escalate_to_human -- a staged tool.

It changes nothing. It returns a structured handoff packet that the caller can
act on, which keeps the "read vs. staged" split honest: no tool in this app
silently mutates state a customer can see.
"""

from typing import Optional

from app.state.conversation_state import ConversationState

# Reasons the assistant is allowed to raise. Free text would let the model
# invent a category, and ops dashboards would be ungroupable.
ESCALATION_REASONS = {
    "lost_parcel_claim",
    "customer_requested_human",
    "customer_frustrated",
    "policy_not_covered",
    "out_of_scope",
    "ambiguous_request",
    "iteration_cap_reached",
    "cross_customer_request",
    "unverified_claim",
}

# Raised by the app itself, never offered to the model -- there is no judgement
# for it to make about whether its own API call failed. Kept out of
# ESCALATION_REASONS so the tool schema stays a list of things a model can
# legitimately choose, but still valid on the wire.
INTERNAL_REASONS = {
    "model_call_failed",
}

VALID_REASONS = ESCALATION_REASONS | INTERNAL_REASONS


def escalate_to_human(
    state: ConversationState,
    reason: str,
    customer_intent: Optional[str] = None,
) -> dict:
    """Stage a handoff. The summary is built from state, never from model prose."""
    normalised = reason if reason in ESCALATION_REASONS else "policy_not_covered"
    state.escalate(normalised)

    summary = state.to_escalation_context()
    summary["customer_intent"] = customer_intent or state.intent
    return {
        "staged": True,
        "action": "handoff_to_human",
        "summary": summary,
        "support_hours": "9:00 AM - 9:00 PM IST, seven days a week",
    }
