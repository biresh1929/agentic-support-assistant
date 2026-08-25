"""Tool schemas and dispatch.

Two structural guarantees live here rather than in a prompt:

1. There is no `apply_discount` tool, and no code path that computes a
   goodwill amount. The model cannot offer a discount because the capability
   does not exist -- not because it was asked nicely not to.
2. Cross-customer lookups are refused in dispatch. Once a session is bound to
   a customer by its first successful lookup, an order belonging to anyone
   else returns an error, whatever the model was persuaded to ask for.
"""

import logging
from typing import Any, Callable

from app.state.conversation_state import ConversationState
from app.tools.escalation import ESCALATION_REASONS, escalate_to_human
from app.tools.eligibility import check_return_eligibility
from app.tools.orders import customer_for_order, get_order_status
from app.tools.policy import search_policy

logger = logging.getLogger(__name__)

# Tools whose first argument identifies an order, and so must pass the
# ownership check before they run.
ORDER_SCOPED_TOOLS = {"get_order_status", "check_return_eligibility"}

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": (
                "Look up one Trendly order by its ID. Returns status, carrier, "
                "tracking, dates and items. Use this before answering anything "
                "about a specific order."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Order ID, e.g. TR-4521.",
                    }
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_return_eligibility",
            "description": (
                "Decide whether an order can be returned or exchanged. This "
                "computes the date arithmetic, category exclusions and final-sale "
                "rules for you. ALWAYS call this instead of working out "
                "eligibility yourself, and narrate what it returns."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "description": "Order ID, e.g. TR-4530."}
                },
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_policy",
            "description": (
                "Search Trendly's shipping and returns policy. Returns the "
                "matching sections with their headings. Call this before "
                "answering ANY question about rules, windows, fees, timelines "
                "or exclusions -- never answer such a question from memory."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to look up, in the customer's own terms.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_human",
            "description": (
                "Hand the conversation to a human agent. Use when the policy "
                "does not cover the question, the customer asks for a human, "
                "the parcel is lost, or you cannot resolve the request."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "enum": sorted(ESCALATION_REASONS),
                        "description": "Why this needs a human.",
                    },
                    "customer_intent": {
                        "type": "string",
                        "description": "One line on what the customer is trying to do.",
                    },
                },
                "required": ["reason"],
            },
        },
    },
]


class OwnershipError(dict):
    """A refusal that is shaped like a tool result, so the loop can narrate it."""


def _check_ownership(state: ConversationState, order_id: str) -> dict | None:
    """Refuse orders belonging to someone other than this session's customer."""
    owner = customer_for_order(order_id)
    if owner is None:
        return None  # a miss is not a leak; get_order_status reports not found
    if state.customer_id and owner != state.customer_id:
        logger.warning(
            "session=%s blocked cross-customer lookup of %s", state.session_id, order_id
        )
        state.record_check("cross_customer_lookup_refused")
        return OwnershipError(
            {
                "error": "order does not belong to this customer",
                "order_id": order_id,
                "guidance": (
                    "Do not reveal any detail of this order. Tell the customer you "
                    "can only discuss orders on their own account, and offer a human "
                    "agent if they believe this is their order."
                ),
            }
        )
    return None


def dispatch(state: ConversationState, name: str, arguments: dict) -> dict:
    """Run one tool call. Every guardrail that must not be bypassable lives here.

    Every tool call in the app goes through here -- the fast path, the agent
    loop and the deterministic escalation tier alike -- which is why the
    per-turn tool record is taken here and not in each caller. A tool recorded
    anywhere else could be missed; one recorded here cannot.
    """
    state.record_tool(name)

    if name in ORDER_SCOPED_TOOLS:
        order_id = str(arguments.get("order_id", ""))
        refusal = _check_ownership(state, order_id)
        if refusal is not None:
            return refusal

    if name == "get_order_status":
        result = get_order_status(str(arguments.get("order_id", "")))
        state.record_check("order_lookup")
        if "error" not in result:
            state.set_active_order(result["order_id"])
            # First successful lookup binds the session. There is no auth on
            # /chat, so this is the strongest available proxy for identity.
            if state.customer_id is None:
                state.customer_id = result["customer_id"]
        return result

    if name == "check_return_eligibility":
        outcome = check_return_eligibility(str(arguments.get("order_id", "")))
        state.record_check("eligibility_check")
        if "error" not in outcome and outcome.get("order_id"):
            state.set_active_order(outcome["order_id"])
        return outcome

    if name == "search_policy":
        chunks = search_policy(str(arguments.get("query", "")))
        state.record_check("policy_search")
        return {"chunks": chunks}

    if name == "escalate_to_human":
        state.record_check("escalation_staged")
        return escalate_to_human(
            state,
            reason=str(arguments.get("reason", "policy_not_covered")),
            customer_intent=arguments.get("customer_intent"),
        )

    logger.error("unknown tool %r requested", name)
    return {"error": "unknown tool", "tool": name}
