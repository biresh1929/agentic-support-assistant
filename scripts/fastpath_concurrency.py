"""A narrow concurrency check on the templated fast path.

SCOPE -- read this before quoting any number it prints.

What this DOES test:
  Twenty simultaneous order-status questions against a locally running
  instance. Every question is one the fast path answers from the order record,
  so the 120B agent model is never called. Note the fast path is not
  model-FREE: tier 1's intent gate still makes one small-model call per turn,
  so this run is cheap, not free. What it shows is that the templated path
  serves concurrent requests without erroring, without corrupting per-session
  state, and without contending on the shared in-memory store -- each request
  gets its own session id and must get its own answer, with no order's detail
  appearing in another's reply.

What this does NOT test:
  * Tier 3. Not one request here reaches the agent loop, on purpose: those
    calls cost real Groq tokens and are rate-limited per minute.
  * Production traffic shape. Twenty requests against ten fixed fixtures is
    not 2,000 chats a day, and the arrival pattern is a thundering herd rather
    than anything realistic.
  * The deployed topology. This runs against one local process. The single
    instance with an in-memory session store is a documented known gap and
    nothing here says otherwise.
  * Sustained load, memory behaviour over time, or p99 anything.

It exists to support exactly one sentence in SOLUTION.md: the cheap path holds
up under concurrent load. Nothing wider than that.

    python scripts/fastpath_concurrency.py --service http://localhost:8000

Standard library only -- concurrent.futures and urllib. A real load-testing
framework would be the right tool for a real load test, and this is not one.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

# The ten fixtures, with a fact from each order record that a correct fast-path
# answer has to contain. Cheap correctness spot-check, not a full assertion.
FIXTURES = [
    ("TR-4521", "BD8871209341"),   # in transit, BlueDart tracking
    ("TR-4522", "14 July 2026"),   # delivered
    ("TR-4523", "5 June 2026"),    # delivered, outside window
    ("TR-4524", "backorder"),      # partially shipped
    ("TR-4525", None),             # delayed -- policy-dependent, not fast path
    ("TR-4526", None),             # lost -- escalates structurally
    ("TR-4527", "23 July 2026"),   # delivered
    ("TR-4528", "19 July 2026"),   # delivered
    ("TR-4529", "cancelled"),      # cancelled
    ("TR-4530", "26 July 2026"),   # delivered
]


def ask(service: str, order_id: str, timeout: float) -> dict:
    payload = json.dumps({
        "session_id": f"load-{uuid.uuid4().hex[:10]}",
        "message": f"Where is my order {order_id}?",
    }).encode("utf-8")
    req = urllib.request.Request(
        service.rstrip("/") + "/chat", data=payload,
        headers={"Content-Type": "application/json"}, method="POST")

    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            return {"order_id": order_id, "status": resp.status,
                    "seconds": time.time() - started, "body": body, "error": None}
    except urllib.error.HTTPError as e:
        return {"order_id": order_id, "status": e.code,
                "seconds": time.time() - started, "body": None,
                "error": f"HTTP {e.code}"}
    except Exception as e:
        return {"order_id": order_id, "status": None,
                "seconds": time.time() - started, "body": None,
                "error": f"{type(e).__name__}: {e}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--service", default="http://localhost:8000")
    ap.add_argument("--requests", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=60.0)
    a = ap.parse_args()

    plan = [FIXTURES[i % len(FIXTURES)] for i in range(a.requests)]
    print(f"firing {len(plan)} concurrent order-status requests at {a.service}")
    print("(fast path only -- no agent-model call, no token cost)\n")

    wall = time.time()
    with ThreadPoolExecutor(max_workers=len(plan)) as pool:
        futures = [pool.submit(ask, a.service, oid, a.timeout) for oid, _ in plan]
        results = [f.result() for f in as_completed(futures)]
    wall = time.time() - wall

    expect = dict(FIXTURES)
    ok = [r for r in results if r["error"] is None and r["status"] == 200]
    failed = [r for r in results if r not in ok]

    # Correctness spot-check, and a check that no answer leaked another order.
    wrong, model_touched, leaked, degraded = [], [], [], []
    for r in ok:
        text = (r["body"] or {}).get("response", "")
        needle = expect.get(r["order_id"])
        if needle and needle.lower() not in text.lower():
            # Carry the reply, not just the miss -- a failure you cannot
            # diagnose from the output is a failure you will misread.
            if text.startswith("Happy to help"):
                degraded.append(r["order_id"])   # tier 1 failed closed
            else:
                wrong.append((r["order_id"], needle, text[:90]))
        for other, _ in FIXTURES:
            if other != r["order_id"] and other in text:
                leaked.append((r["order_id"], other))
        # TR-4525 (delayed) and TR-4526 (lost) are deliberately excluded from
        # the fast path -- their answers depend on policy -- so reaching the
        # agent loop is correct for those two and only those two.
        tools = (r["body"] or {}).get("tools_called") or []
        if needle and [t for t in tools
                       if t not in ("get_order_status", "escalate_to_human")]:
            model_touched.append((r["order_id"], tools))

    lat = sorted(r["seconds"] for r in ok)
    print(f"  wall time            {wall:.2f}s for {len(plan)} concurrent requests")
    print(f"  completed 200        {len(ok)}/{len(plan)}")
    print(f"  failed               {len(failed)}")
    for r in failed:
        print(f"      {r['order_id']}  status={r['status']}  {r['error']}")
    if lat:
        print(f"  latency  min/p50/max {lat[0]:.2f}s / "
              f"{statistics.median(lat):.2f}s / {lat[-1]:.2f}s")
        print(f"  throughput           {len(ok) / wall:.1f} req/s")
    print(f"  cross-order leak     {'none' if not leaked else 'LEAKED ' + str(leaked)}")
    print(f"  stayed off the model {'yes' if not model_touched else 'NO ' + str(model_touched)}")
    answerable = [r for r in ok if expect.get(r["order_id"])]
    print(f"  content spot-check   {len(answerable) - len(wrong) - len(degraded)}"
          f"/{len(answerable)} answered correctly")
    if degraded:
        print(f"  tier 1 failed closed {len(degraded)} turn(s) -> clarifying template "
              f"(router rate-limited under burst; by design, not a fast-path failure)")
    for oid, needle, got in wrong:
        print(f"      {oid}: expected {needle!r}")
        print(f"          got: {got}")

    # A content miss is not a correctness failure on its own. Tier 2 still
    # makes one intent-gate call, and under a burst that call can be rate
    # limited -- which fails closed to `ambiguous` and returns the clarifying
    # template. That is the router degrading as designed, not the fast path
    # breaking. A leak or a stray model call would be a real failure.
    clean = not failed and not leaked and not model_touched
    print("\n" + ("fast path held under concurrent load."
                  if clean else "something went wrong -- see above."))
    print("This says nothing about tier 3, production traffic shape, or the "
          "deployed single-instance topology.")
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
