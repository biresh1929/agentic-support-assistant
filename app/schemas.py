from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=4000)


class ChatResponse(BaseModel):
    """What one turn produced, and enough evidence to audit it.

    The last three fields are observability, not decoration. `escalated: true`
    with no machine-readable reason cannot be routed to a queue, and a claim
    about the return window is only trustworthy if you can see that
    check_return_eligibility was actually called rather than guessed at. All
    three are scoped to the turn being answered, not the conversation.
    """

    response: str
    escalated: bool = False
    session_id: str
    tools_called: list[str] = Field(default_factory=list)
    # Set only when this turn escalated; one of app.tools.escalation.VALID_REASONS.
    escalation_reason: str | None = None
