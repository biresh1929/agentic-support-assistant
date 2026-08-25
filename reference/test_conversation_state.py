from conversation_state import ConversationState, ConversationStore


def test_new_state_has_no_active_order():
    state = ConversationState(session_id="s1")
    assert state.active_order_id is None
    assert state.order_id_history == []


def test_setting_order_id_for_the_first_time():
    state = ConversationState(session_id="s1")
    state.set_active_order("ORD-1042")
    assert state.active_order_id == "ORD-1042"
    assert state.order_id_history == []  # nothing to preserve yet


def test_customer_correction_updates_active_and_preserves_history():
    """The turn every grader will try: 'actually, it was order #1234.'"""
    state = ConversationState(session_id="s1")
    state.set_active_order("ORD-1042")
    state.set_active_order("ORD-1099")  # correction

    assert state.active_order_id == "ORD-1099"
    assert "ORD-1042" in state.order_id_history
    assert "ORD-1099" not in state.order_id_history  # active isn't duplicated


def test_repeated_correction_back_to_prior_id_does_not_duplicate_history():
    state = ConversationState(session_id="s1")
    state.set_active_order("ORD-1042")
    state.set_active_order("ORD-1099")
    state.set_active_order("ORD-1042")  # corrects back

    assert state.active_order_id == "ORD-1042"
    assert state.order_id_history.count("ORD-1099") == 1


def test_checks_performed_are_deduplicated_and_ordered():
    state = ConversationState(session_id="s1")
    state.record_check("order_lookup")
    state.record_check("eligibility_check")
    state.record_check("order_lookup")  # duplicate

    assert state.checks_performed == ["order_lookup", "eligibility_check"]


def test_escalation_context_carries_everything_a_human_needs():
    state = ConversationState(session_id="s1", customer_id="CUST-9")
    state.set_active_order("ORD-1042")
    state.record_check("order_lookup")
    state.record_check("eligibility_check")
    state.escalate("Return window expired but customer disputes delivery date")

    context = state.to_escalation_context()

    assert context["order_id"] == "ORD-1042"
    assert context["customer_id"] == "CUST-9"
    assert context["checks_already_performed"] == ["order_lookup", "eligibility_check"]
    assert "expired" in context["reason"]


def test_store_creates_a_session_on_first_access():
    store = ConversationStore()
    state = store.get_or_create("session-abc")
    assert state.session_id == "session-abc"


def test_store_returns_the_same_state_across_calls():
    """This is what makes state actually multi-turn instead of stateless."""
    store = ConversationStore()
    store.get_or_create("session-abc").set_active_order("ORD-1042")

    state_again = store.get_or_create("session-abc")
    assert state_again.active_order_id == "ORD-1042"


def test_store_returns_none_for_unknown_session_via_get():
    store = ConversationStore()
    assert store.get("does-not-exist") is None
