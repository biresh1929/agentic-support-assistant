"""FastAPI surface: a health check and the single /chat endpoint.

Routing is tiered on purpose. Most of Trendly's volume is repetitive status
checks where the order record is the whole answer, so those never reach a
70B-class model. See SOLUTION.md for the cost argument.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import get_settings, today
from app.retrieval.index import get_index
from app.router import agent_loop, fast_path
from app.router.intent_gate import classify
from app.schemas import ChatRequest, ChatResponse
from app.state.conversation_state import ConversationState, ConversationStore
from app.tools.registry import dispatch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the retrieval index before the first request, not during it.

    get_index() is lru_cached, so whichever request touched it first used to
    pay the whole build. On a small instance that was slow enough that the
    first policy question blocked until the platform decided the service was
    unhealthy and restarted it -- taking the process down instead of answering.
    Paying it here means the health check does not pass until retrieval is
    actually ready, which is what gates traffic correctly.
    """
    index = get_index()
    logger.info("retrieval index ready: %s chunks", len(index.chunks))
    yield


app = FastAPI(title="Trendly Support Assistant", version="0.2.0", lifespan=lifespan)

# In-memory, per-process. Swapping in Redis means replacing this line only --
# ConversationState itself has no storage dependency.
store = ConversationStore()

ASK_FOR_ORDER = (
    "Happy to check that for you — which order is it? You'll find the order "
    "number in your confirmation email; it looks like TR-4521."
)
ASK_FOR_DETAIL = (
    "Happy to help — could you tell me a bit more about what you need? If it's "
    "about a specific order, the order number is the fastest way in."
)
OUT_OF_SCOPE = (
    "I can only help with Trendly orders, shipping, returns, refunds and "
    "exchanges. I'll pass you to a human colleague for anything else."
)
FRUSTRATED = (
    "I'm sorry — that's genuinely frustrating, and I don't want to make you "
    "repeat yourself. I'm passing you to a human colleague now with everything "
    "we've covered so far."
)
LOST_PARCEL = (
    "I'm sorry — the carrier has marked {order_id} as lost in transit. Under "
    "policy 1.6 that's a lost-parcel claim rather than a return, and claims "
    "like this are resolved by a human colleague rather than by me. I'm "
    "passing it to one now, with everything we've covered so far. They'll "
    "arrange either a free replacement or a full refund, whichever you prefer."
)

# Statuses whose correct answer depends on policy, so tier 2 declines them.
POLICY_DEPENDENT_STATUSES = {"delayed", "lost_in_transit"}


@app.get("/health")
def health() -> dict:
    settings = get_settings()
    return {
        "status": "ok",
        "reference_date": today().isoformat(),
        "orders_loaded": settings.orders_path.exists(),
        "policy_loaded": settings.policy_path.exists(),
        "llm_configured": bool(settings.groq_api_key),
        "router_model": settings.router_model,
        "agent_model": settings.agent_model,
    }


def _stage(state: ConversationState, reason: str, intent: str) -> None:
    """Stage a handoff the app decided on itself, rather than the model.

    Routed through dispatch for the same reason the model's escalations are:
    dispatch is the one place that records the tool call and the check, so a
    deterministic escalation shows up in the transcript identically to a
    model-chosen one.
    """
    dispatch(state, "escalate_to_human", {"reason": reason, "customer_intent": intent})


def _reply(state: ConversationState, text: str, *, escalated: bool) -> ChatResponse:
    """Build the wire response from what this turn actually did.

    The escalation reason is read from the per-turn slot rather than the
    sticky one, so a conversation that escalated at turn 2 does not keep
    reporting that reason on turns 3 and 4.
    """
    return ChatResponse(
        response=text,
        escalated=escalated,
        session_id=state.session_id,
        tools_called=list(state.turn_tools),
        escalation_reason=state.turn_escalation_reason if escalated else None,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    state = store.get_or_create(request.session_id)
    state.next_turn()

    routing = classify(request.message, state)
    previous_intent, state.intent = state.intent, routing.intent
    logger.info(
        "session=%s turn=%s intent=%s order_id=%s",
        state.session_id, state.turn_count, routing.intent, routing.order_id,
    )

    if routing.intent == "order_status":
        return _handle_order_status(routing.order_id, state, request.message)

    if routing.intent in {"policy_question", "eligibility_check"}:
        if routing.order_id:
            state.set_active_order(routing.order_id)
        return _run_agent(request.message, state)

    return _handle_escalation_tier(routing, state, previous_intent, request.message)


def _handle_order_status(
    order_id: str | None, state: ConversationState, message: str
) -> ChatResponse:
    """Tier 2. The order record answers the question, so no model is called."""
    target = order_id or state.active_order_id
    if not target:
        state.set_pending_question("which order are you asking about?")
        return _reply(state, ASK_FOR_ORDER, escalated=False)

    order = dispatch(state, "get_order_status", {"order_id": target})

    if order.get("error") == "order does not belong to this customer":
        # Staged explicitly: this branch escalates, so it owes the handoff a
        # routable reason. Without one the ticket lands in a queue of "escalated,
        # cause unknown", which is where cross-customer attempts least belong.
        dispatch(
            state,
            "escalate_to_human",
            {"reason": "cross_customer_request",
             "customer_intent": "asked about an order on another account"},
        )
        return _reply(
            state,
            "I can only look up orders on your own account, so I can't share "
            "anything about that one. If you believe it's yours, I'll put you "
            "through to a human colleague who can verify it.",
            escalated=True,
        )

    if "error" in order:
        return _reply(state, fast_path.render_not_found(target), escalated=False)

    state.set_pending_question(None)

    if order.get("requires_human"):
        # Structural, not advisory. get_order_status has flagged this since it
        # was written, but nothing consumed the flag, so the handoff happened
        # only when the model happened to call escalate_to_human that turn.
        # The harness caught the failure mode that leaves: the same question
        # produced "I'll forward this to a human colleague" with escalated
        # False and no tool call on one run, and a correct handoff on the next.
        # A promise of a human with no ticket behind it is worse than a
        # refusal, so the decision is taken here where it cannot vary.
        #
        # HUMAN_ONLY_STATUSES is currently exactly {lost_in_transit}; adding a
        # status to it means revisiting this reason, which is why the mapping
        # is spelled out rather than defaulted.
        dispatch(
            state,
            "escalate_to_human",
            {"reason": "lost_parcel_claim",
             "customer_intent": "parcel marked lost in transit by the carrier"},
        )
        return _reply(
            state,
            LOST_PARCEL.format(order_id=order["order_id"]),
            escalated=True,
        )

    if fast_path.can_fast_path(order):
        return _reply(state, fast_path.render(order), escalated=False)

    # delayed / lost_in_transit: the answer needs policy, so tier 3 handles it.
    logger.info("status=%s escalating from fast path to agent loop", order["status"])
    return _run_agent(message, state)


def _run_agent(message: str, state: ConversationState) -> ChatResponse:
    """Tier 3."""
    result = agent_loop.run(message, state)
    logger.info(
        "session=%s agent used %s tool call(s): %s",
        state.session_id, result.tool_calls_made, result.tools_used,
    )
    # Whether the assistant just asked something is decided by what it actually
    # said, not by which tools ran. The prompt tells it to ask for a missing
    # size in prose rather than calling the staging tool and letting that fail,
    # so the tool's needs_info never fires on that path and cannot be the only
    # signal. Next turn the router is told a question is outstanding, which is
    # what keeps a bare "size L please" from being read as a new, vague request.
    if result.text.strip().endswith("?"):
        if not state.pending_question:
            state.set_pending_question("the detail it just asked the customer for")
    else:
        state.set_pending_question(None)
    return _reply(state, result.text, escalated=result.escalated)


def _handle_escalation_tier(
    routing, state, previous_intent: str | None, message: str
) -> ChatResponse:
    """Tier 4. Deterministic -- there is nothing here worth a model call."""
    if routing.intent == "ambiguous":
        # Ask once. A second vague turn means clarifying is not working.
        if previous_intent != "ambiguous":
            state.set_pending_question("what does the customer need?")
            return _reply(state, ASK_FOR_DETAIL, escalated=False)
        _stage(state, "ambiguous_request", "unclear request")
        return _reply(
            state,
            ASK_FOR_DETAIL.split("—")[0] + "— I'll bring in a human colleague to "
            "help work out what you need.",
            escalated=True,
        )

    if routing.intent == "out_of_scope":
        _stage(state, "out_of_scope", "off-topic request")
        return _reply(state, OUT_OF_SCOPE, escalated=True)

    # escalation_trigger. These go through the agent loop rather than a canned
    # line: "my parcel is lost", "I want a human" and "give me 20% off" all land
    # here, and a single template answers none of them well. The loop stages the
    # handoff itself with the right reason. The template is the fallback.
    result = agent_loop.run(message, state)
    if not result.text:
        _stage(state, "customer_frustrated", "wants a human")
        return _reply(state, FRUSTRATED, escalated=True)
    if not result.escalated:
        _stage(state, "customer_requested_human", "wants a human")
    return _reply(state, result.text, escalated=True)


def _status_of(order_id: str) -> str | None:
    from app.tools.orders import get_order_status

    return get_order_status(order_id).get("status")
