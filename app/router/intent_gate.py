"""Tier 1: a small, cheap model call that decides where a message goes.

This is a model call rather than keyword rules on purpose -- "it never showed
up" and "where is it" carry no shared keyword, and rules over phrasing break
the moment a customer types something unanticipated.
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional

from app.config import get_settings
from app.llm import get_client
from app.prompts.intent_gate import INTENT_GATE_SYSTEM, context_hint
from app.state.conversation_state import ConversationState

logger = logging.getLogger(__name__)

INTENTS = {
    "order_status",
    "policy_question",
    "eligibility_check",
    "escalation_trigger",
    "out_of_scope",
    "ambiguous",
}


@dataclass
class Routing:
    intent: str
    order_id: Optional[str] = None
    confidence: float = 0.0


def classify(message: str, state: ConversationState) -> Routing:
    """Classify one message. Any failure routes to `ambiguous`, which escalates.

    Failing closed matters: a router that guesses `order_status` when the model
    is unavailable would answer confidently from a half-built context.
    """
    settings = get_settings()
    try:
        completion = get_client().chat.completions.create(
            model=settings.router_model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": INTENT_GATE_SYSTEM},
                {
                    "role": "system",
                    "content": context_hint(
                        state.active_order_id, state.intent, state.pending_question
                    ),
                },
                {"role": "user", "content": message},
            ],
        )
        payload = json.loads(completion.choices[0].message.content or "{}")
    except Exception:
        logger.exception("intent gate failed; routing to ambiguous")
        return Routing(intent="ambiguous", confidence=0.0)

    intent = payload.get("intent")
    if intent not in INTENTS:
        logger.warning("intent gate returned unknown intent %r", intent)
        return Routing(intent="ambiguous", confidence=0.0)

    order_id = payload.get("order_id")
    if not isinstance(order_id, str) or not order_id.strip():
        order_id = None

    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0

    return Routing(intent=intent, order_id=order_id, confidence=confidence)
