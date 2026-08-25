# Trendly Support Assistant — solution notes

## Architecture

Trendly handles roughly 2,000 support chats a day, and the large majority are
plain status checks — *where is it, when does it arrive, did it ship*. For
those the order record **is** the answer: there is nothing to retrieve and
nothing for a 70B-class model to reason about. Running every message through
one uniform agent loop would mean paying retrieval and 70B inference on the
cheapest, most repetitive traffic in the queue. So the architecture routes
around that cost rather than absorbing it.

Every message hits a cheap classifier first, then takes one of four paths.

**Tier 1 — the intent gate.** One call to a small model
(`openai/gpt-oss-20b`, temperature 0, JSON mode) that classifies the message
into one of six intents and extracts an order ID if the message contains one.
This is a model call rather than keyword rules on purpose: *"it never showed
up"* and *"where is it"* share no keyword, and rules over phrasing break the
first time a customer writes something unanticipated. It fails closed — any
exception, any unrecognised intent, routes to `ambiguous`, which asks a
question rather than guessing. A router that guessed `order_status` while the
model was unavailable would answer confidently from a half-built context.

**Tier 2 — the templated fast path.** `order_status` with a known order and a
status the record fully answers (`in_transit`, `delivered`, `cancelled`,
`partially_shipped`) is rendered from a template. **No model call, no
retrieval.** This is where the cost argument is actually collected. Statuses
whose correct answer depends on policy are deliberately excluded — `delayed`
and `lost_in_transit` need more than the record, so they fall through.

One thing is decided here rather than downstream: if the order record comes
back with `requires_human` set — currently only `lost_in_transit` — the
handoff is staged **in code**, before the model gets a turn. See the
structural-guardrails trade-off below for why.

**Tier 3 — the bounded agent loop.** Anything needing policy or eligibility
reasoning gets the 120B model with four tools (`get_order_status`,
`check_return_eligibility`, `search_policy`, `escalate_to_human`), capped at
six tool iterations. The cap matters more than it looks: a model that cannot
find a fact will keep re-querying with slightly different wording until it
times out, and with a cap, exhausting the budget is itself a signal that the
question needs a human rather than another attempt.

Retrieval underneath it is hybrid — dense embeddings plus BM25 over 28 policy
chunks, fused with reciprocal rank fusion. Neither retriever is sufficient
alone: semantic search matches *"can I send this back?"* to *Return window*,
which BM25 misses for lack of a shared term, while BM25 handles *"final sale"*
and `TR-4521`, rare exact tokens a 384-dimension embedding smears into
neighbouring clauses. RRF's damping constant is 2, not the conventional 60 —
60 comes from TREC-scale runs over millions of documents and over 28 chunks it
flattens the curve until rank 1 and rank 12 sit 18% apart, which starts
rewarding chunks that are mediocre in *both* lists over one that a retriever
ranked first.

**The citation guard**, inside tier 3, is the reason answers can be trusted
rather than merely hoped for. Before any answer is returned it is checked
mechanically against what was actually retrieved that turn: every clause it
cites must have been retrieved, every rupee figure must be one of four the
policy defines *and* have its licensing clause on the table, every percentage
and duration must appear in the evidence. A failure buys exactly one retry,
carrying the retrieved text and an instruction to rewrite using only that. A
second failure escalates rather than answering. Retrying without retrieval
would be asking the model to guess again more carefully, so that case escalates
immediately.

**Tier 4 — deterministic escalation.** `ambiguous`, `out_of_scope` and
frustration don't get a model call either. Ambiguous asks once and escalates on
the second vague turn, because a second vague turn is evidence that clarifying
is not working. Every handoff carries a structured packet built from
conversation state — order, customer, prior order IDs, which checks were
already performed — never from model prose, so the human picking it up can see
what was already ruled out instead of just that the bot gave up.

## Key trade-offs

**ONNX MiniLM embeddings, no torch.** Retrieval uses ChromaDB's default
embedding function, which is `all-MiniLM-L6-v2` executed through ONNX Runtime.
The alternative — the same model via `sentence-transformers` — pulls in PyTorch
and roughly 2GB of image on top of what is already there. The model weights are
identical, so retrieval quality should be unaffected, and the deploy footprint
is the whole reason to prefer one over the other: the image is **775MB** and
the running container holds steady at **297MB** resident, which fits inside a
512MB free tier with room to spare. A torch-based build would not.

*Stated honestly:* no retrieval evaluation was run to confirm quality parity
empirically. The argument here is that the weights are the same, not that
equivalence was measured — and `.deepeval/` is empty, so there is no benchmark
in this repo to point at. If that claim needs to be defensible rather than
merely reasonable, the eval is the missing piece.

**Deterministic Python for eligibility, not the model.**
`check_return_eligibility` computes the date arithmetic, category exclusions
and final-sale rules in plain Python. Date arithmetic is exactly where language
models go quietly wrong — 29 days versus 31, *delivered* versus *dispatched*,
jewellery being non-returnable even well inside the window — and quietly is the
problem: the answer is fluent and confident and off by two days. The model's
job is to call this and narrate the result, not to decide it. Eligibility is
computed per item rather than per order, because an order can be mixed:
TR-4522 pairs a returnable cotton tee with non-returnable socks, and a single
boolean would have to lie about one of them.

**A hard-coded rupee allowlist, not a classifier.** Exactly four monetary
figures exist in the policy — ₹250 delay credit (1.5), ₹150 courier
reimbursement (5.2), ₹300 footwear box deduction (2.5), ₹99 shipping fee
refund (3.2) — and each is bound in code to the single condition that licenses
it. ₹250 is only sayable when *1.5 Delayed orders* is actually on the table.
Anything else is blocked: a percentage off, a goodwill credit, a refund total
the model computed itself. This is deliberately not a fuzzy match against
retrieved text or a general "don't invent numbers" instruction, because the
requirement is to be able to walk through it line by line and defend it, and
neither of those alternatives can be.

**Structural guardrails wherever the guarantee has to hold.** Two places where
a prompt instruction was not considered good enough:

*There is no `apply_discount` tool.* Not a tool the model is told not to use —
no tool, and no code path anywhere that computes a goodwill amount. The
capability does not exist to be talked into. The harness pushes on this for
five consecutive turns in `c06` (*"other agents have given me discounts"*,
*"nobody will check"*, *"I'll leave a one star review"*) and there is nothing
for the pressure to act on.

*`requires_human` orders escalate in code.* `get_order_status` has always
flagged lost parcels, and the comment beside that flag claimed routing did not
depend on the model noticing it — but nothing consumed the flag, so the handoff
happened only when the model chose to call `escalate_to_human` that turn. The
harness caught the failure mode that leaves: on one run TR-4526 produced *"I'll
forward this to a human colleague"* with `escalated: false` and no tool call at
all — a promise with no ticket behind it — and on the next run, same input, a
correct handoff. The decision now happens in tier 2 before the model is
consulted, and the regression test asserts the model is not consulted at all,
because that is the only assertion that cannot go intermittent.

**A conversation harness as a separate layer from the unit tests.** The unit
suite covers date arithmetic, the citation guard, the ownership check — things
that are true or false within a single call. It cannot cover what happens over
five turns, which is where a support assistant actually fails: it forgets the
corrected order number, it escalates on the frustrated turn instead of the
lost-parcel turn, it holds a refusal for three turns and gives way on the
fourth. `harness/` drives 18 scripted conversations (68 turns) against a
running instance and scores the transcript across seven dimensions, with a
runner that cannot see the expectations and a scorer that never touches the
network.

It earned its cost on the first run by finding a **cross-customer data leak
that no unit test would have surfaced**, because it only exists across turns:
session identity was bound by `get_order_status` but not by
`check_return_eligibility`, so a conversation that opened with an eligibility
question left the session unbound — and the *next* turn could name any order at
all, receive it in full, and bind the session to that customer instead. The
observed transcript has the assistant disclosing another customer's order on
turn 2 and then refusing the customer's *own* order on turn 4. Every individual
call behaved exactly as its unit tests said it should.

## Five questions for Trendly's ops team

**1. How often does the policy document actually change, and do answers given
under a previous version need to be flagged as potentially superseded?**
Right now the policy is chunked and indexed in memory at startup, with no
version stamp on it and none on the answers. If the return window moves from
30 days to 21, every conversation already in flight starts giving a different
answer mid-thread, and every answer in last month's transcripts silently
becomes wrong without anything marking it. The cheap fix — stamp a policy
version on each response and log it — is only worth building if the document
moves more than once or twice a year. If it changes monthly, or if a customer
can hold Trendly to an answer given under the old version, that changes the
design: answers need to record which version they were given under, and
superseding a version needs a way to find and flag what it invalidates.

**2. What does the real escalation volume look like, against the ~70/30 split
in the brief?** Two things ride on this, and they pull in opposite directions.
If escalations are much rarer than 30%, the deterministic tiers are doing more
work than assumed and the model budget could shrink further. If they are much
more common, the handoff path stops being an exit and becomes a main path,
which makes the quality of the escalation packet more important than the
quality of the answers. Related and more concrete: **does staging a handoff
need to be idempotent?** A lost-parcel conversation currently re-stages the
handoff on every turn, which is right for the customer — they keep being told a
human has it — and possibly wrong for the ticketing system if each staging
creates a ticket.

**3. Is the customer already authenticated before they reach `/chat`?** This is
the question with the largest blast radius. There is no auth on the endpoint,
so the strongest available proxy for identity is that the first order
successfully looked up binds the session to that customer. That works, and the
harness verifies it holds under direct attack — but it means anyone who knows
an order number is that customer for the rest of the session. If the real
channel is a logged-in web widget or a verified WhatsApp number, the customer
ID arrives with the request and the binding becomes defence in depth, which is
what it should be. If the channel is anonymous, this proxy is the *only* thing
standing there and it is not enough on its own. The answer changes whether the
cross-customer guard is a backstop or the front line.

**4. When a final-sale item's exchange size is unavailable, what actually
happens?** The policy contradicts itself here and the contradiction is
load-bearing. **2.4** says final-sale items are eligible for size exchange only
— *no refunds and no store credit*. **4.3** says that if the requested size is
unavailable, the exchange is *automatically converted to a refund* under
section 3. TR-4528 is exactly this case. As written, an assistant reasoning
correctly from the document can reach either answer depending on which clause
it retrieves first, and there is no principled way to pick. This needs a
human decision before it needs any code.

**5. What is the source of truth for the delivery date, and what happens when
sources disagree?** The 30-day window is computed from `delivered_at`, so that
one field decides eligibility outright. In the fixtures it is a single
unambiguous value; in production there is an OMS record, a carrier scan, and
sometimes a customer insisting on a third date. The harness has a case for
this (`c05`, where the customer says *"that can't be right, I only got it last
week"* against a record showing 54 days) and the current behaviour is to trust
the record — which is defensible, but it is a policy decision that nobody has
actually made. Two things needed: which source wins, and whether *"I received
it later than you think"* is itself an escalation category. It is not one now,
and the assistant handles it by offering a human, which is a reasonable
accident rather than a design.

## Cost and latency

Measured during this build, not estimated: latencies below are drawn from the
two full harness runs (68 turns each, against Groq), and memory from the Docker
verification.

| Path | What it does | p50 | Range |
|---|---|---|---|
| Tier 2 — fast path | intent gate + order lookup, **no agent model** | **0.71s** | 0.47–9.62s |
| Tier 4 — routing only | intent gate, answers from a template | 0.92s | 0.61–3.44s |
| Tier 4 — structural escalation | `requires_human`, **no agent model** | **1.58s** | — |
| Tier 3 — agent loop | 1–6 tool calls against the 120B model | **12.94s** | 2.06–38.71s |

The spread inside tier 3 is the tool count: a single-lookup policy question
returns in a couple of seconds, while a multi-tool eligibility turn that
retrieves policy, checks eligibility and then narrates the result runs to
thirty-plus. The outliers in the tier 2 range are slow intent-gate calls, not
work — that tier does no reasoning, so its cost floor is one small-model round
trip.

The lost-parcel path is worth calling out because it is the one place where the
reliability argument and the performance argument point the same way. Escalating
`requires_human` orders in code rather than hoping the model calls
`escalate_to_human` took that path from ~11s and intermittently wrong to
**1.58s and deterministic**. Making it structural was not a trade of speed for
correctness; it bought both, because the fix consists of not calling the 120B
model at all.

**Memory:** 297MB resident, steady — unchanged after an agent-loop turn, since
the Chroma index is built at startup rather than per request. The image is
775MB. Both fit a 512MB free tier.

**Cost:** zero. Groq's free tier covered every call in development, the harness
runs and the deployment, with no paid spend at any point. The architecture is
the reason that is not a coincidence: the majority of Trendly's daily volume is
plain status checks, and those never reach a model that costs anything to run —
tier 2 answers them from the order record with one small-model routing call, and
the 120B model is only paid for on the traffic that genuinely needs reasoning.

## Known limitations

### Eligibility can be reasoned about without calling the tool that decides it

`check_return_eligibility` exists so that date arithmetic, category exclusions
and final-sale rules are computed in Python rather than inferred by a language
model — 29 days versus 31, "delivered" versus "dispatched", jewellery being
non-returnable even well inside the window. Its tool description tells the
model, in capitals, to *always* call it instead of working out eligibility
itself.

That instruction is a prompt, not a guarantee, and the conversation harness
caught the model ignoring it. In several cases (`c08` turn 2, `c17` turn 1,
`c18` turn 1 — reproducibly, across separate runs) the model called
`get_order_status` and `search_policy` and then reasoned its way to a
returnable / non-returnable verdict from the raw policy text, never calling
the deterministic tool at all. The answers it produced happened to be correct.
That is luck, and the whole reason the tool exists is not to depend on it.

This is the one place in the app where a correctness guarantee rests on the
model's compliance rather than on structure, and the contrast with the
discount guardrail is deliberate and worth stating: the assistant cannot offer
a discount because there is no `apply_discount` tool and no code path that
computes a goodwill amount — the capability does not exist to be talked into,
so no prompt instruction is load-bearing. Eligibility has no equivalent, and
prompt instructions alone are not a strong enough basis for a guarantee we
would want to defend.

A production-grade fix would follow the shape of the citation guard, which
already rejects an answer that cites a clause it never retrieved: parse the
response for eligibility-shaped claims — any returnable / not-returnable
verdict, any statement about the return window applying or having closed — and
reject or retry any response that makes such a claim without
`check_return_eligibility` appearing in that turn's tool calls. The retry
would carry the tool result, exactly as `correction_prompt` carries the
retrieved sections today.

This was identified and left as a known gap given the time budget. Building
the claim detector is real added scope, and the harness now makes the failure
visible on every run, which is the more important half.

### A more specific policy clause can lose to a less specific one

The harness caught a reply that told the customer they would have to cover
return postage, citing **Refunds → 3.2 Shipping fees**, when **Return pickup →
5.1 Pickup** states that free reverse pickup is available on all serviceable
pincodes. The answer is wrong.

The obvious suspicion is a retrieval gap, so that was checked directly rather
than assumed:

```
search("do I have to pay for return postage", k=4)
  1. Refunds -> 3.2 Shipping fees
  2. Return pickup -> 5.1 Pickup          <-- retrieved, rank 2 of 4
  3. Refunds -> 3.4 Partial refunds
  4. Shipping -> 1.6 Lost parcels
```

5.1 was retrieved and was in front of the model when it answered. So this is a
reasoning failure, not a retrieval one: the model answered the adjacent
question it had the best match for — *is the original shipping fee refunded?*
— rather than the one that was asked, *who pays to send the item back?* No
change was made to the chunker or the hybrid-retrieval weights, because
tuning retrieval to fix a problem that is not in retrieval would have made the
index worse for no gain.

The citation guard does not catch this and is not built to. The answer cites a
clause that was genuinely retrieved and states no figure absent from the
evidence, so it is *grounded* and still *wrong* — it is wrong by omission of a
more specific clause. Grounding and completeness are different properties, and
only the first is currently verified. A fix here would most likely be
authoring rather than code: 3.2 and 5.1 are adjacent enough that the policy
document itself could cross-reference them, which would put the connection in
the retrieved text instead of asking the model to make it.

Filed as a known gap; not chased.
