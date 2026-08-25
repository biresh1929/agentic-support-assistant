from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class ConversationState:
    session_id: str
    customer_id: Optional[str] = None
    active_order_id: Optional[str] = None
    order_id_history: list[str] = field(default_factory=list)
    intent: Optional[str] = None
    pending_question: Optional[str] = None
    checks_performed: list[str] = field(default_factory=list)
    turn_count: int = 0
    escalation_reason: Optional[str] = None
    # Per-turn, cleared by next_turn(). checks_performed and escalation_reason
    # are cumulative because the handoff packet wants the whole conversation;
    # the /chat response wants only what happened on the turn it is answering.
    # Reporting the cumulative flags on the wire would mean every turn after
    # the first escalation claims to have escalated too.
    turn_tools: list[str] = field(default_factory=list)
    turn_escalation_reason: Optional[str] = None
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def set_active_order(self, order_id: str) -> None:
        """Update the active order, handling corrections.

        Last-write-wins: if the customer corrects themselves mid-conversation
        ("actually it was order 1099, not 1042"), the new ID becomes active
        and the old one moves into history rather than being discarded --
        so a later "go back to my first order" is still answerable.
        """
        if self.active_order_id and self.active_order_id != order_id:
            if self.active_order_id not in self.order_id_history:
                self.order_id_history.append(self.active_order_id)
        self.active_order_id = order_id
        self.updated_at = datetime.now(timezone.utc)

    def record_check(self, check: str) -> None:
        """Log a check the agent has already performed this conversation.

        This is what makes an escalation summary useful instead of a
        rephrased "I don't know" -- it lets the handoff say what was
        already ruled out, not just that the agent gave up.
        """
        if check not in self.checks_performed:
            self.checks_performed.append(check)
        self.updated_at = datetime.now(timezone.utc)

    def record_tool(self, name: str) -> None:
        """Log a tool call made on the current turn.

        Not deduplicated, unlike record_check: two lookups in one turn is a
        different (and more expensive) thing than one, and the harness reads
        this to check that the agent called a tool rather than answering from
        memory.
        """
        self.turn_tools.append(name)
        self.updated_at = datetime.now(timezone.utc)

    def set_pending_question(self, question: Optional[str]) -> None:
        self.pending_question = question

    def escalate(self, reason: str) -> None:
        self.escalation_reason = reason
        self.turn_escalation_reason = reason

    def to_escalation_context(self) -> dict:
        """Everything a human agent needs to pick this up without re-asking."""
        return {
            "session_id": self.session_id,
            "customer_id": self.customer_id,
            "order_id": self.active_order_id,
            "prior_order_ids": self.order_id_history,
            "intent": self.intent,
            "checks_already_performed": self.checks_performed,
            "reason": self.escalation_reason,
            "turns_in_conversation": self.turn_count,
        }

    def next_turn(self) -> None:
        self.turn_count += 1
        self.turn_tools = []
        self.turn_escalation_reason = None


class ConversationStore:
    """In-memory session store keyed by session_id.

    Swap this for a LangGraph checkpointer or Redis later without touching
    ConversationState itself -- the dataclass above has no framework
    dependency on purpose, so it isn't locked to whichever agent loop
    (LangGraph vs. a hand-rolled ReAct loop) you end up wiring it into.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, ConversationState] = {}

    def get_or_create(self, session_id: str) -> ConversationState:
        if session_id not in self._sessions:
            self._sessions[session_id] = ConversationState(session_id=session_id)
        return self._sessions[session_id]

    def get(self, session_id: str) -> Optional[ConversationState]:
        return self._sessions.get(session_id)
