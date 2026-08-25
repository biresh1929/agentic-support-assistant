"""The conversation runner.

Delivers turns and records evidence. It knows nothing about whether an answer
was right -- all of that lives in score_conversations.py, and the two meet at
a transcript file. You can re-score an old run without re-running it, and you
cannot tune the runner to help the score.

    python harness/run_conversations.py --service http://localhost:8000 \
                                        --cases harness/conversations.json \
                                        --out runs/latest

Standard library only, so it runs against a deployed container with nothing
installed alongside it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

# The agent loop is allowed six tool iterations, each a Groq round trip, so a
# turn that legitimately does the work can take a while. These are wall-clock
# ceilings on pathology, not latency targets.
TURN_DEADLINE_S = 45.0
CASE_DEADLINE_S = 300.0
HEALTH_WAIT_S = 60


# ---------------------------------------------------------------------------
# ADAPTER. The only two functions that know this service's wire format.
# ---------------------------------------------------------------------------

def build_request(session_id: str, text: str) -> dict:
    """/chat takes a session and a message. Note what is absent: there is no
    customer_id on the wire. Identity is bound by the first successful order
    lookup (app/tools/registry.py), so a case establishes who it is by which
    order its opening turn names -- which is itself the thing the
    cross-customer cases are probing.
    """
    return {"session_id": session_id, "message": text}


def normalise(body: object) -> dict:
    """Map ChatResponse onto the shape the scorer reads.

    A missing field is recorded as missing and scored as a contract violation,
    never quietly defaulted to something that would pass.
    """
    if not isinstance(body, dict):
        return {"_unparseable": True}
    return {
        "reply": body.get("response"),
        "session_id": body.get("session_id"),
        "tools_called": body.get("tools_called"),
        "escalated": body.get("escalated"),
        "escalation_reason": body.get("escalation_reason"),
        "raw": body,
    }

# ---------------------------------------------------------------------------


def _post(url: str, payload: dict, timeout: float) -> tuple[int, object]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", "replace")
        try:
            return resp.status, json.loads(raw)
        except json.JSONDecodeError:
            return resp.status, {"_unparseable": raw[:2000]}


def wait_healthy(service: str, seconds: int) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(service.rstrip("/") + "/health", timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def runner_view(case: dict) -> dict:
    """The case with every expectation stripped out.

    Cases and their expectations live in one file because authoring eighteen
    multi-turn scripts across two files is miserable, so the separation the
    two-file layout would give for free is enforced here instead. Only `say`
    reaches the wire; `customer_id`, `expect` and `case_checks` are scoring
    metadata and this function is the only thing standing between them and
    the service.
    """
    return {
        "case_id": case["case_id"],
        "session_id": case.get("session_id"),
        "reuses_session_of": case.get("reuses_session_of"),
        "turns": [{"say": t["say"]} for t in case["turns"]],
    }


def run_case(
    service: str,
    case: dict,
    sessions: dict,
    turn_deadline: float,
    case_deadline: float,
    quiet: bool,
) -> dict:
    """Drive one conversation to the end, failures included.

    A failed turn does not abort the case. Escalation triggers are usually the
    last turn, and aborting on the first timeout throws away the evidence the
    case exists to collect. Later turns are marked after_failure so the scorer
    can discount them rather than blindly crediting them.
    """
    view = runner_view(case)

    borrowed = view.get("reuses_session_of")
    if borrowed:
        if borrowed in sessions:
            session_id = sessions[borrowed]
        else:
            # --only, or a case ordered before the one it borrows from.
            print(f"    ! {view['case_id']} reuses the session of {borrowed}, "
                  f"which has not run; using a fresh session instead")
            session_id = f"conv-{view['case_id']}-{uuid.uuid4().hex[:8]}"
    else:
        session_id = view.get("session_id") or \
            f"conv-{view['case_id']}-{uuid.uuid4().hex[:8]}"
    sessions[view["case_id"]] = session_id

    url = service.rstrip("/") + "/chat"
    case_t0 = time.time()
    turns_out: list[dict] = []
    degraded = False

    for idx, turn in enumerate(view["turns"], start=1):
        remaining = case_deadline - (time.time() - case_t0)
        row = {"turn": idx, "say": turn["say"], "after_failure": degraded}

        if remaining <= 0:
            row.update({"http_status": None, "error": "case deadline exhausted",
                        "latency_s": 0.0, "in_deadline": False,
                        "skipped": True, "response": None})
            turns_out.append(row)
            degraded = True
            if not quiet:
                print(f"    SKIP t{idx} (case deadline)", flush=True)
            continue

        budget = min(turn_deadline, remaining)
        t0 = time.time()
        status, body, err = None, None, None
        try:
            status, body = _post(url, build_request(session_id, turn["say"]), budget)
        except urllib.error.HTTPError as e:
            status, err = e.code, f"HTTP {e.code}"
            try:
                body = json.loads(e.read().decode("utf-8"))
            except Exception:
                body = None
        except Exception as e:
            # Every exception is swallowed into `err` and recorded. One wedged
            # turn can never stall the rest of the run.
            err = f"{type(e).__name__}: {e}"
        dt = time.time() - t0

        # Decided here, at capture time, not re-derived from latency later.
        in_deadline = err is None and status == 200 and dt <= budget
        if not in_deadline:
            degraded = True

        row.update({"http_status": status, "error": err,
                    "latency_s": round(dt, 3), "in_deadline": in_deadline,
                    "skipped": False,
                    "response": normalise(body) if body is not None else None})
        turns_out.append(row)
        if not quiet:
            tag = "ok  " if in_deadline else "MISS"
            print(f"    {tag} t{idx} {dt:6.2f}s  {turn['say'][:56]}", flush=True)

    return {
        "case_id": view["case_id"],
        "session_id": session_id,
        "reused_session_of": borrowed,
        "case_latency_s": round(time.time() - case_t0, 3),
        "completed": all(t["in_deadline"] for t in turns_out),
        "turns": turns_out,
    }


def run(
    service: str,
    cases: list,
    outdir: Path,
    turn_deadline: float = TURN_DEADLINE_S,
    case_deadline: float = CASE_DEADLINE_S,
    quiet: bool = False,
) -> list:
    outdir.mkdir(parents=True, exist_ok=True)
    if not wait_healthy(service, HEALTH_WAIT_S):
        raise SystemExit(f"service at {service} never became healthy")

    sessions: dict[str, str] = {}
    transcript = []
    for case in cases:
        if not quiet:
            print(f"  {case['case_id']}  ({len(case['turns'])} turns)", flush=True)
        transcript.append(
            run_case(service, case, sessions, turn_deadline, case_deadline, quiet)
        )

    with (outdir / "transcript.jsonl").open("w", encoding="utf-8", newline="\n") as fh:
        for row in transcript:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return transcript


def load_cases(path: Path) -> list:
    text = path.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("//")]


def main() -> None:
    # Windows consoles default to cp1252, and policy answers contain the rupee
    # sign. Losing a whole run to a console encoding error is a silly way to
    # lose a run.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser()
    ap.add_argument("--service", default="http://localhost:8000")
    ap.add_argument("--cases", default="harness/conversations.json")
    ap.add_argument("--out", default="runs/latest")
    ap.add_argument("--turn-deadline", type=float, default=TURN_DEADLINE_S)
    ap.add_argument("--case-deadline", type=float, default=CASE_DEADLINE_S)
    ap.add_argument("--only", default=None,
                    help="run one case_id, for iterating on a single script")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cases = load_cases(Path(args.cases))
    if args.only:
        cases = [c for c in cases if c["case_id"] == args.only]
        if not cases:
            raise SystemExit(f"no case with case_id {args.only!r}")

    transcript = run(args.service, cases, Path(args.out), args.turn_deadline,
                     args.case_deadline, args.quiet)

    turns = sum(len(c["turns"]) for c in transcript)
    landed = sum(1 for c in transcript for t in c["turns"] if t["in_deadline"])
    whole = sum(1 for c in transcript if c["completed"])
    print(f"\n{landed}/{turns} turns answered inside the deadline; "
          f"{whole}/{len(transcript)} conversations ran to the end.")
    print("Completion is not behaviour. Score the transcript to find out "
          "whether the conversations went the way they were meant to.")


if __name__ == "__main__":
    main()
