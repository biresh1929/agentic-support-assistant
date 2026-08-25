"""Checks on the conversation harness itself.

The harness is only worth trusting if two things hold: it cannot leak an
expected answer into a prompt, and its idea of a valid response matches the
app's. Neither is self-evident from reading either file alone, so both are
asserted here.

Nothing in this module talks to the network or the model.
"""

import importlib.util
import json
from pathlib import Path

import pytest

from app.schemas import ChatResponse
from app.tools.escalation import ESCALATION_REASONS, VALID_REASONS

HARNESS = Path(__file__).resolve().parent.parent / "harness"


def _load(name: str):
    """Import a harness script by path; harness/ is scripts, not a package."""
    spec = importlib.util.spec_from_file_location(name, HARNESS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load("run_conversations")
scorer = _load("score_conversations")
CASES = json.loads((HARNESS / "conversations.json").read_text(encoding="utf-8"))
CASE_IDS = {c["case_id"] for c in CASES}


# --------------------------------------------------------------------------
# The runner must not be able to leak the answer key
# --------------------------------------------------------------------------

def test_runner_view_strips_every_expectation():
    """Only `say` reaches the wire.

    The single-question harness gets this for free by keeping questions and
    the answer key in separate files. Conversations are authored in one file
    for sanity, so runner_view() is the only thing enforcing the separation --
    which makes it the one function here that must never quietly grow a field.
    """
    for case in CASES:
        view = runner.runner_view(case)
        serialised = json.dumps(view)
        assert "expect" not in serialised, case["case_id"]
        assert "case_checks" not in serialised, case["case_id"]
        assert "customer_id" not in serialised, case["case_id"]
        assert all(set(t) == {"say"} for t in view["turns"]), case["case_id"]


def test_request_body_carries_only_session_and_message():
    assert set(runner.build_request("s1", "hello")) == {"session_id", "message"}


# --------------------------------------------------------------------------
# The scorer's copy of the enum must not drift from the app's
# --------------------------------------------------------------------------

def test_scorer_escalation_enum_matches_the_app():
    """The scorer keeps a literal copy rather than importing this set.

    A scorer that imports its definition of "valid" from the code under test
    cannot notice that code widening the enum -- so the copy is deliberate and
    this test is what keeps it honest. If this fails, update
    score_conversations.VALID_ESCALATION_REASONS to match, and check that
    every reason_in in conversations.json still means what it did.
    """
    assert scorer.VALID_ESCALATION_REASONS == VALID_REASONS


def test_internal_reasons_are_not_offered_to_the_model():
    from app.tools.registry import TOOL_SCHEMAS

    escalate = next(t for t in TOOL_SCHEMAS
                    if t["function"]["name"] == "escalate_to_human")
    offered = set(escalate["function"]["parameters"]["properties"]["reason"]["enum"])
    assert offered == ESCALATION_REASONS
    assert "model_call_failed" not in offered


def test_chat_response_reports_what_the_turn_did():
    fields = ChatResponse.model_fields
    assert {"response", "escalated", "session_id",
            "tools_called", "escalation_reason"} <= set(fields)


# --------------------------------------------------------------------------
# The cases file must be internally consistent
# --------------------------------------------------------------------------

def test_case_ids_are_unique():
    ids = [c["case_id"] for c in CASES]
    assert len(ids) == len(set(ids))


def test_every_turn_has_something_to_say():
    for case in CASES:
        assert case["turns"], case["case_id"]
        for i, turn in enumerate(case["turns"], start=1):
            assert turn.get("say", "").strip(), f"{case['case_id']} t{i}"


def test_borrowed_sessions_point_at_a_real_case_that_runs_first():
    order = [c["case_id"] for c in CASES]
    for case in CASES:
        borrowed = case.get("reuses_session_of")
        if borrowed is None:
            continue
        assert borrowed in CASE_IDS, case["case_id"]
        assert order.index(borrowed) < order.index(case["case_id"]), (
            f"{case['case_id']} borrows a session from {borrowed}, which runs later"
        )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["case_id"])
def test_escalation_expectations_are_satisfiable(case):
    """A typo'd reason makes a check impossible to pass, and looks like a bug
    in the app rather than a bug in the case."""
    esc = (case.get("case_checks") or {}).get("escalation")
    if not esc:
        return
    assert esc.get("expect") in {"never", "once", "at_least_once"}, case["case_id"]
    for reason in esc.get("reason_in", []):
        assert reason in VALID_REASONS, f"{case['case_id']}: unknown reason {reason!r}"
    if esc.get("expect") == "never":
        assert "reason_in" not in esc and "not_before_turn" not in esc, (
            f"{case['case_id']}: 'never' plus a reason or a turn bound is contradictory"
        )


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["case_id"])
def test_turn_indices_stay_inside_the_conversation(case):
    checks = case.get("case_checks") or {}
    n = len(case["turns"])

    for per in checks.get("persisted", []):
        assert 1 <= per["from_turn"] <= n, case["case_id"]
        assert 1 <= per["through_turn"] <= n, case["case_id"]
        assert per["from_turn"] < per["through_turn"], case["case_id"]
    for sup in checks.get("superseded", []):
        # A correction on the final turn leaves no later turn to observe it in.
        assert 1 <= sup["after_turn"] < n, case["case_id"]
    for key in ("refusal_sticky", "session_binding"):
        block = checks.get(key)
        if block and "from_turn" in block:
            assert 1 <= block["from_turn"] <= n, f"{case['case_id']} {key}"
    esc = checks.get("escalation") or {}
    if esc.get("not_before_turn"):
        assert 1 < esc["not_before_turn"] <= n, case["case_id"]


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["case_id"])
def test_per_turn_expectations_line_up_with_turns(case):
    for i, turn in enumerate(case["turns"], start=1):
        expect = turn.get("expect")
        if expect is None:
            continue
        unknown = set(expect) - {"must_contain", "must_not_contain", "tools_called",
                                 "tools_not_called", "escalated", "escalation_reason"}
        assert not unknown, f"{case['case_id']} t{i}: unknown keys {unknown}"
        if "escalation_reason" in expect:
            assert expect["escalation_reason"] in VALID_REASONS, case["case_id"]
            assert expect.get("escalated") is True, (
                f"{case['case_id']} t{i}: a reason without escalated=true is a "
                f"contract violation, so the case would be asserting a bug"
            )


# --------------------------------------------------------------------------
# The property that cost a rewrite to notice
# --------------------------------------------------------------------------

def test_a_service_that_answers_nothing_scores_nothing():
    """Silence satisfies every negative check unless checks are windowed.

    Before scoring was gated on "did any turn in this check's window actually
    answer", an empty transcript scored 48/100 -- full marks on safety, and
    most of scope_isolation -- because a service that says nothing never says
    anything wrong.
    """
    card = scorer.score_run(CASES, [])
    assert card["quality_total_score"] == 0.0
    assert all(v == 0.0 for v in card["dimensions"].values())
    # Every case still appears: dropping one must not shrink the denominator.
    assert len(card["cases"]) == len(CASES)


def test_a_leak_zeroes_the_whole_case_not_just_the_turn():
    case = next(c for c in CASES
                if (c.get("case_checks") or {}).get("never_mentions"))
    leaked = case["case_checks"]["never_mentions"][0]
    session = "s-leak"
    transcript = [{
        "case_id": case["case_id"],
        "session_id": session,
        "turns": [{"turn": i + 1, "in_deadline": True,
                   "response": {"reply": leaked if i == 0 else "fine",
                                "session_id": session, "tools_called": [],
                                "escalated": False, "escalation_reason": None}}
                  for i in range(len(case["turns"]))],
    }]
    scored = scorer.score_run(case_subset := [case], transcript)["cases"][0]
    assert scored["leak"] == [leaked]
    assert sum(scored["earned"].values()) == 0.0
    assert sum(scored["available"].values()) > 0.0
    assert case_subset  # the case was scored, not skipped
