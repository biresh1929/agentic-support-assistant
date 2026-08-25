"""Tier 3: a bounded tool-calling loop for anything the fast path cannot answer.

The cap matters more than it looks. Without it a model that cannot find a fact
will keep re-querying with slightly different wording until it times out; with
it, exhausting the budget is itself a signal that the question needs a human.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from app.config import get_settings
from app.guardrails.citation_guard import correction_prompt, verify
from app.llm import get_client
from app.prompts.system import build_system_prompt
from app.state.conversation_state import ConversationState
from app.tools.registry import TOOL_SCHEMAS, dispatch

logger = logging.getLogger(__name__)


@dataclass
class AgentResult:
    text: str
    escalated: bool = False
    tool_calls_made: int = 0
    tools_used: list[str] = field(default_factory=list)
    # Policy text retrieved this turn. The citation guard checks answers
    # against exactly this, so it is collected as the loop runs.
    retrieved_chunks: list[dict] = field(default_factory=list)
    hit_cap: bool = False
    # Every tool result this turn, so the citation guard can tell a figure
    # grounded in an order record from one the model invented.
    evidence: list[str] = field(default_factory=list)
    guard_retried: bool = False
    guard_failed: bool = False


UNGROUNDED_MESSAGE = (
    "I don't have that in Trendly's policy, and I don't want to guess at it. "
    "I'm passing you to a human colleague who can confirm properly."
)
CAP_MESSAGE = (
    "I haven't been able to pin this down from Trendly's policy, so I'm passing "
    "you to a human colleague who can look at it properly."
)
ERROR_MESSAGE = (
    "Something went wrong on my side just now. Let me hand you to a human "
    "colleague so you're not stuck."
)


def _assistant_turn(message: Any) -> dict:
    """Serialise an assistant message with tool calls back into the transcript."""
    return {
        "role": "assistant",
        "content": message.content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in message.tool_calls
        ],
    }


def run(message: str, state: ConversationState) -> AgentResult:
    settings = get_settings()
    result = AgentResult(text="")

    messages: list[dict] = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": message},
    ]
    if state.active_order_id:
        messages.insert(
            1,
            {
                "role": "system",
                "content": (
                    f"The order under discussion is {state.active_order_id}. "
                    "Look it up rather than assuming its contents."
                ),
            },
        )

    for iteration in range(settings.max_tool_iterations):
        try:
            completion = get_client().chat.completions.create(
                model=settings.agent_model,
                temperature=0.2,
                tools=TOOL_SCHEMAS,
                tool_choice="auto",
                messages=messages,
            )
        except Exception:
            logger.exception("agent loop model call failed")
            state.escalate("model_call_failed")
            return AgentResult(text=ERROR_MESSAGE, escalated=True, hit_cap=False)

        reply = completion.choices[0].message

        if not reply.tool_calls:
            answer = (reply.content or "").strip()
            verdict = verify(answer, result.retrieved_chunks, result.evidence)

            if verdict.ok:
                result.text = answer
                # Per-turn, not cumulative. checks_performed never forgets, so
                # reading it here made every turn after the first escalation
                # report escalated=True -- turning one handoff into a stream of
                # them, and making "escalated exactly once" unobservable.
                result.escalated = state.turn_escalation_reason is not None
                return result

            logger.warning(
                "session=%s grounding check failed: %s",
                state.session_id, verdict.summary(),
            )

            # Exactly one retry, and only if a retry could help: with nothing
            # retrieved there is nothing to correct the answer against.
            if result.guard_retried or not result.retrieved_chunks:
                state.escalate("unverified_claim")
                state.record_check("escalation_staged")
                result.text = UNGROUNDED_MESSAGE
                result.escalated = True
                result.guard_failed = True
                return result

            result.guard_retried = True
            state.record_check("citation_guard_retry")
            messages.append({"role": "assistant", "content": answer})
            messages.append(
                {
                    "role": "system",
                    "content": correction_prompt(verdict, result.retrieved_chunks),
                }
            )
            continue

        messages.append(_assistant_turn(reply))

        for call in reply.tool_calls:
            try:
                arguments = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                logger.warning("unparseable tool arguments: %r", call.function.arguments)
                arguments = {}

            outcome = dispatch(state, call.function.name, arguments)
            result.tool_calls_made += 1
            result.tools_used.append(call.function.name)
            result.evidence.append(json.dumps(outcome, ensure_ascii=False, default=str))
            if call.function.name == "search_policy":
                result.retrieved_chunks.extend(
                    outcome.get("chunks", []) if isinstance(outcome, dict) else []
                )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(outcome, ensure_ascii=False),
                }
            )

    # Budget exhausted without a final answer.
    logger.info("session=%s hit the %s-call cap", state.session_id, settings.max_tool_iterations)
    state.escalate("iteration_cap_reached")
    state.record_check("escalation_staged")
    result.text = CAP_MESSAGE
    result.escalated = True
    result.hit_cap = True
    return result
