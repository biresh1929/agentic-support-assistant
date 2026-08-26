# Trendly Support Assistant

A customer support assistant for a direct-to-consumer fashion retailer. It
answers questions about orders, shipping, returns, refunds and exchanges from
Trendly's own order records and policy document — and hands off to a human when
it should, rather than guessing.

`POST /chat` is the API, `GET /health` reports readiness, and `GET /` serves a
small chat page for trying it by hand.

- **[SOLUTION.md](SOLUTION.md)** — architecture, trade-offs, questions for ops, known limitations
- **[PROMPTS.md](PROMPTS.md)** — both prompts in full, and what changed in them and why
- **[harness/README.md](harness/README.md)** — the conversation-level test harness

## Live deployment

**https://trendly-assistant.onrender.com**

Open that in a browser and you get the chat UI, not JSON — `GET /` serves a
single-file demo page. It is the quickest way to see the thing work: each
assistant turn shows the tools it called as small pills, so a fast-path answer
visibly uses one tool and a staged return uses four, and any turn that hands
off is marked with its escalation reason. The API is unchanged underneath.

```bash
curl https://trendly-assistant.onrender.com/health

curl -X POST https://trendly-assistant.onrender.com/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo","message":"can I return the kurta from TR-4530?"}'
```

Deployed from [`render.yaml`](render.yaml). `GROQ_API_KEY` is set in Render's
dashboard and marked `sync: false`, so it is never stored in the repo or baked
into the image.

**Note for reviewers.** Render's free tier sleeps after 15 minutes idle, so the
first request after a quiet spell takes up to a minute while the container
wakes and builds the retrieval index. Everything after that is normal speed —
roughly a second for an order-status lookup, and ten to thirty for a policy or
eligibility question, which is several Groq round trips on 0.1 of a CPU. Render
was chosen over Railway because Railway's free tier is a one-time trial credit
that expires, and this needs to stay reachable for two weeks.

## Running it

Either path needs a Groq API key. Copy `.env.example` to `.env` and fill in
`GROQ_API_KEY`. `.env` is gitignored and dockerignored; it never reaches an
image.

### Docker

```bash
docker build -t trendly-assistant .
docker run -p 8000:8000 --env-file .env trendly-assistant
```

The container reads all configuration from the environment at runtime and
listens on `$PORT` (default 8000), so the same image runs locally and on a
platform that assigns the port dynamically. It runs as a non-root user, holds
about 176MB resident, and shuts down cleanly on SIGTERM.

### Local development

```bash
python -m venv .venv && .venv/Scripts/activate     # Windows
# python3 -m venv .venv && source .venv/bin/activate   # macOS / Linux

pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

### The demo page

With the server running, open **http://localhost:8000/**. One static HTML
file, no build step and no dependencies — it posts to the same `/chat` the
curl examples below use, and renders each turn's `tools_called` and
`escalation_reason` so the routing is visible while you talk to it.

### Checking it works

```bash
curl localhost:8000/health

curl -X POST localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"session_id":"demo","message":"Where is my order TR-4521?"}'
```

```json
{
  "response": "Order TR-4521 is on its way. It's with BlueDart under tracking number BD8871209341, and is expected to arrive on 31 July 2026. ...",
  "escalated": false,
  "session_id": "demo",
  "tools_called": ["get_order_status"],
  "escalation_reason": null
}
```

`tools_called` and `escalation_reason` are there so a turn can be audited: an
`escalated: true` with no machine-readable reason cannot be routed to a queue,
and a claim about the return window is only trustworthy if you can see that
`check_return_eligibility` was actually called rather than guessed at.

Reuse a `session_id` across requests to continue a conversation. The first
order successfully looked up binds that session to its customer; any other
customer's order is refused from then on.

### Tests

```bash
python -m pytest                       # 175 unit + contract tests, no network

# conversation harness -- needs a running instance and a real API key
python harness/run_conversations.py   --service http://localhost:8000 --out runs/latest
python harness/score_conversations.py --transcript runs/latest/transcript.jsonl
```

The harness is a separate layer, not more unit tests: 18 scripted multi-turn
conversations scored across seven behavioural dimensions. It exists because the
failures that matter in a support assistant happen *between* turns — forgetting
a corrected order number, escalating on the wrong turn, giving way on the fourth
push after refusing three times. See [harness/README.md](harness/README.md).

## Try these

The dataset is ten fixed orders across four customers, with the clock pinned to
2026-07-29 so the dates stay meaningful.

| Ask about | What should happen |
|---|---|
| `TR-4521` — where is it? | Templated answer, no model call at all |
| `TR-4530` — can I return the kurta? | In window and returnable, so the return is **raised**, not just described |
| `TR-4527` — return the earrings? | Refused on **category** (jewellery, 2.3), not on date |
| `TR-4523` — return the jacket? | Refused on **date** (54 days, past 2.1) |
| `TR-4528` — refund the shirt? | Final sale: refund refused, and it **asks which size** you want instead (2.4) |
| `TR-4526` — where is it? | Lost parcel: escalates in code before the model is consulted |
| Ask for 30% off | Declined — there is no discount tool to invoke |
| Ask about another customer's order | Refused, and the session stays bound to you |
| Reply `size L` after it asks | The bare answer continues the return, and the exchange is staged |

## A note on AI assistance

Claude Code was used heavily on this, and the honest split is not "AI wrote it"
or "AI helped a bit" — it is that the design decisions were mine and most of the
typing was not.

**Mine — the decisions:**

- **The tiered routing design.** Recognising that most of the volume is status
  checks that need no model and no retrieval, and that the right response is to
  route around that cost rather than run everything through one uniform agent
  loop. The four-tier split, and which statuses belong in the templated path
  versus the agent loop, were design calls made against the cost argument.
- **The citation-guard retry.** Deciding that grounding should be *verified
  mechanically* rather than instructed, that a failure gets exactly one retry
  carrying the retrieved text, and that a failure with nothing retrieved should
  escalate immediately — because retrying with no evidence is just asking the
  model to guess again more carefully.
- **Structural guardrails over prompt instructions.** The judgement that where a
  guarantee has to hold it belongs in code, not in prose: no `apply_discount`
  tool exists at all, the rupee allowlist is four hard-coded amounts each bound
  to its licensing clause, and `requires_human` orders escalate before the model
  gets a turn. Each of those was a decision to spend more effort for a guarantee
  that cannot be talked out of.
- **Building the conversation harness, and what to do with what it found.**
  Deciding that unit tests could not reach the failures that actually matter,
  that the runner must not be able to see the expectations, and — when it
  surfaced a cross-customer leak, a fabricated handoff, and an eligibility tool
  being bypassed — which of those to fix structurally and which to write up as
  known limitations rather than chase before the deadline.

**Claude Code's — the execution:**

- Implementation of essentially all of it from those decisions: the FastAPI
  surface, the hybrid retriever, the chunker, the eligibility rules, the tool
  registry, the harness runner and scorer.
- Debugging, and some of it genuinely collaborative rather than dictated —
  tracking the cross-customer leak down to session binding living in the
  `get_order_status` branch but not the `check_return_eligibility` one; working
  out that the RRF constant of 60 was flattening rank signal across only 28
  chunks; finding that a scorer bug was awarding 48/100 to a service that
  answered nothing, because negative checks pass vacuously against silence.
- **Working the harness's findings through to fixed and verified.** Four real
  bugs were caught and closed this way, each landing as its own commit with a
  regression test and each re-checked against the live deployment rather than
  only asserted: a fabricated handoff on lost parcels (the assistant promised a
  human with no ticket behind it, on one run and not the next), cross-customer
  session binding (an opening eligibility check left the session unbound, so the
  next turn could name any order and receive it in full), an eligibility-tool
  bypass (verdicts reasoned from raw policy text instead of the deterministic
  tool), and a pending-question routing gap (a bare *"size L please"* classified
  as ambiguous, so the exchange never staged). The last of those took two
  attempts — the first fix passed its harness case by luck and failed a direct
  call, which is in PROMPTS.md.
- Test writing, docstrings, and these documents — drafted from the decisions
  and the evidence above, then reviewed and corrected by me.

**A note on how I actually think about this.** I believe AI should
be used as an assistant, not a crutch -- if you're nothing without
it, you probably didn't need it in the first place. The person who
gets the most out of AI is the one who could still do the job
without it, just slower.

One trick I use to catch hallucination: I tell Claude Code, in its
own instructions, to open every session with "Hello, Biresh" -- a
canary. If that greeting is ever missing or off, something upstream
broke and I stop trusting the output until I've checked. I have a
few habits like this; it's less a trick and more how I work as
someone who builds AI-native, not someone who just happens to use
AI.

I also came in with a design instinct AI didn't give me. Before
touching this assignment, I already knew the first move is figuring
out who I'm actually talking to and what problem they're really
describing -- not the one they stated. That's from being an AI/ML
intern before this, where the lesson that stuck was: listening
and earning trust matters more than technical output, and the real
problem is rarely the one you're told at the start.

Where I used AI and where I didn't: I used AI for implementation,
debugging, and drafting docs. Not for deciding the architecture, and
not for deciding when something was done. I care about quality
versus completeness -- a first version has to work and be complete
before it gets polished, and left alone AI will happily
over-engineer past that line. I stayed the one deciding when to
stop.

AI also can't take ownership, and this role demands full ownership
from discovery through delivery -- that's not something a tool can
carry. Same with compliance, data leakage, and auditability:
those calls were mine, not something a model flagged for me. I
designed the tests with AI's help, but I'm the one who checked
whether they were actually good, not just present.

So yes, I used AI throughout. But I also thoroughly read, checked and questioned the code it wrote -- that's the difference this should show.

Where this document states a measured number, it was measured in this repo.
Two claims that could have been asserted are deliberately absent because they
could not be: there is no retrieval evaluation backing the embedding choice,
and three of the four prompt iterations in PROMPTS.md reconstruct their
before-text rather than quoting commits, because those edits predate `git init`.
Both are flagged where
they appear.
