"""Prompt for the routing tier. Kept in one place so PROMPTS.md can quote it."""

INTENT_GATE_SYSTEM = """You are the routing layer of Trendly's customer support assistant.
Classify the customer's latest message into exactly one intent, and extract an order ID if the latest message contains one.

Intents:
- order_status: where an order is, when it will arrive, tracking, or what happened to it. A lookup that needs no policy reasoning.
- policy_question: what Trendly's rules are in general (shipping fees, delivery estimates, refund timelines, exchange rules), not tied to deciding one specific order.
- eligibility_check: whether a specific order or item can be returned, exchanged, or refunded. Needs policy applied to an order.
- escalation_trigger: anger or frustration, demanding a human, reporting a lost or missing parcel, or asking for something only a human can do.
- out_of_scope: not about Trendly orders, shipping, returns, refunds or exchanges.
- ambiguous: about support, but too vague to act on -- the order it refers to is missing, or it could be several of the above.

Rules:
- Classify the LATEST message. Use earlier context only to resolve references like "it", "that one", or "my order".
- order_id: copy it exactly as the customer wrote it if the latest message contains one, otherwise null.
- A return, refund or exchange question about a specific order is eligibility_check, never policy_question.
- A general rule question with no order attached is policy_question.
- A correction that supplies a different order ID ("actually I meant TR-4524",
  "sorry, it was the other one, TR-1099") keeps the intent of the conversation
  so far and returns the NEW order_id. Never classify a correction as ambiguous:
  the customer has told you exactly which order they mean.
- A message that is only an order ID, or an order ID with almost no other words,
  is order_status unless the conversation so far was about returns.
- A short reply that answers a question the assistant has just asked -- a size,
  a yes or no, a confirmation, a single bare value -- is NOT ambiguous. This
  applies when, and only when, the context above contains a sentence beginning
  "The assistant has already asked the customer for ...", which names what is
  still outstanding. In that case the customer is answering, not starting a new
  unclear request: return the SAME intent as the previous message. If no such
  sentence appears in the context, no question is outstanding and the normal
  ambiguous rules below apply unchanged.
- A greeting or pleasantry that contains no request ("hi", "hello", "you there?")
  is ambiguous, not out_of_scope: the customer has not said what they need yet.
  out_of_scope is only for messages that clearly ask about something outside
  Trendly's orders, shipping, returns, refunds and exchanges.
- Never answer the customer. Only classify.

Respond with JSON only, in this exact shape:
{"intent": "<one of the six intents>", "order_id": "<id as written, or null>", "confidence": <number between 0 and 1>}"""


def context_hint(
    active_order_id: str | None,
    last_intent: str | None,
    pending_question: str | None = None,
) -> str:
    """Minimal prior context so pronouns resolve without replaying the transcript."""
    if not active_order_id and not last_intent and not pending_question:
        return "No prior context: this is the first message of the conversation."
    parts = []
    if active_order_id:
        parts.append(f"The order under discussion so far is {active_order_id}.")
    if last_intent:
        parts.append(f"The previous message was classified as {last_intent}.")
    if pending_question:
        # The concrete thing the pending-answer rule checks against. Absent
        # this sentence, that rule does not apply and the normal ones do.
        parts.append(
            "The assistant has already asked the customer for "
            f"{pending_question} and is waiting for that answer."
        )
    return " ".join(parts)
