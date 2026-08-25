"""Checks a policy answer against what was actually retrieved this turn.

The model is asked to ground every policy claim. This verifies it did, rather
than trusting it. Three checks, all mechanical:

1. Every clause the answer cites (2.1, 3.3, ...) must be among the clauses
   retrieved this turn. Citation style is normalised first -- the model drifts
   between bracket and paren forms across calls even at low temperature, and
   that is cosmetic, not a grounding failure.
2. Every specific figure -- days, hours, percentages, rupees -- must appear in
   the evidence: the retrieved policy text plus this turn's tool results. An
   order delivered "15 days ago" is grounded in the order record, not the
   policy, so both count as evidence.
3. Every rupee figure must be one of the four amounts the policy defines, AND
   the clause that licenses it must have been retrieved. ₹250 is only sayable
   when 1.5 Delayed orders is on the table.

A figure inside a refusal ("I can't give you 20% off") is not a claim, so a
short negation window before the number suppresses the finding.
"""

import json
import re
from dataclasses import dataclass, field

from app.guardrails.amounts import ALLOWED_AMOUNTS, ALLOWED_RUPEE_VALUES

# "Returns -> 2.1 ...", "(policy 2.3)", "[2.4]", "Section 3.1" -- all reduce to
# the clause number, which is the part that must be verifiable.
CLAUSE_RE = re.compile(r"(?<!\d)(\d{1,2}\.\d{1,2})(?!\d)")
CURRENCY_RE = re.compile(r"(?:₹|Rs\.?\s?|INR\s?)\s?([\d,]+)", re.IGNORECASE)
# A bare number followed by a money word is still a monetary claim. Without
# this, "250 store credit" walks straight past a check that only looks for ₹.
BARE_AMOUNT_RE = re.compile(
    r"(?<![\d.])([\d,]{2,})\s*(?:rupee|rupees|store credit|credit|refund|"
    r"reimbursement|deduction|voucher|discount|fee|charge|off\b)",
    re.IGNORECASE,
)
PERCENT_RE = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*(?:%|percent)", re.IGNORECASE)
DURATION_RE = re.compile(
    r"(?<!\d)(\d{1,3})\s*(?:calendar\s+|business\s+|working\s+)?"
    r"(day|days|hour|hours|week|weeks|month|months)\b",
    re.IGNORECASE,
)
NEGATION_RE = re.compile(
    r"(can'?t|cannot|can not|unable|not able|no longer|don'?t|do not|never|"
    r"not authoris|not authoriz|isn'?t|aren'?t|without)",
    re.IGNORECASE,
)
# How far back to look for a negation before treating a figure as a claim.
NEGATION_WINDOW = 60


@dataclass
class GuardVerdict:
    ok: bool
    violations: list[str] = field(default_factory=list)
    retrieved_clauses: set[str] = field(default_factory=set)

    def summary(self) -> str:
        return "; ".join(self.violations)


def _is_negated(text: str, position: int) -> bool:
    return bool(NEGATION_RE.search(text[max(0, position - NEGATION_WINDOW) : position]))


def _mentions_number(evidence: str, number: str) -> bool:
    return bool(re.search(rf"(?<!\d){re.escape(number)}(?!\d)", evidence))


def _clauses_in(text: str) -> set[str]:
    return set(CLAUSE_RE.findall(text))


def verify(answer: str, retrieved_chunks: list[dict], evidence: list[str]) -> GuardVerdict:
    """Check one answer. `evidence` is every tool result produced this turn."""
    if not answer.strip():
        return GuardVerdict(ok=True)

    chunk_text = "\n".join(c.get("text", "") for c in retrieved_chunks)
    evidence_text = "\n".join(evidence)
    haystack = chunk_text + "\n" + evidence_text

    # A clause is grounded by retrieval OR by a tool that resolved it in Python.
    # check_return_eligibility decides 2.3 deterministically and reports it in
    # policy_basis; treating search_policy as the only evidence would falsely
    # escalate almost every eligibility answer that cites its own basis.
    retrieved_clauses = (
        _clauses_in(chunk_text)
        | _clauses_in(evidence_text)
        | {c["clause"].split()[0] for c in retrieved_chunks if c.get("clause")}
    )

    violations: list[str] = []

    # 1. cited clauses must have been retrieved
    for clause in sorted(_clauses_in(answer)):
        if clause not in retrieved_clauses:
            violations.append(
                f"cites policy {clause}, which was not retrieved this turn"
            )

    # 2. rupee figures: allowlisted, and licensed by a retrieved clause
    money = list(CURRENCY_RE.finditer(answer)) + list(BARE_AMOUNT_RE.finditer(answer))
    for match in money:
        raw = match.group(1).replace(",", "")
        if not raw.isdigit() or _is_negated(answer, match.start()):
            continue
        value = int(raw)
        if value not in ALLOWED_RUPEE_VALUES:
            violations.append(f"states ₹{value}, which the policy does not define")
            continue
        allowed = next(a for a in ALLOWED_AMOUNTS if a.rupees == value)
        if allowed.citation.rsplit("-> ", 1)[-1].split()[0] not in retrieved_clauses:
            violations.append(
                f"states ₹{value} without retrieving {allowed.citation}, "
                f"which is the only clause that licenses it"
            )

    # 3. percentages -- the policy contains none, so any claim is invented
    for match in PERCENT_RE.finditer(answer):
        if _is_negated(answer, match.start()):
            continue
        if not _mentions_number(haystack, match.group(1)):
            violations.append(f"states {match.group(1)}%, which appears nowhere in policy")

    # 4. durations must appear in the evidence
    for match in DURATION_RE.finditer(answer):
        number = match.group(1)
        if not _mentions_number(haystack, number):
            violations.append(
                f"states '{match.group(0).strip()}', which is not in the retrieved text"
            )

    # de-duplicate while preserving order
    seen: set[str] = set()
    unique = [v for v in violations if not (v in seen or seen.add(v))]
    return GuardVerdict(ok=not unique, violations=unique, retrieved_clauses=retrieved_clauses)


def correction_prompt(verdict: GuardVerdict, retrieved_chunks: list[dict]) -> str:
    """The single retry nudge: what went wrong, and what is actually available."""
    sections = "\n".join(
        f"- {c['citation']}: {c['text'][:400]}" for c in retrieved_chunks
    ) or "- (nothing was retrieved)"
    return (
        "Your previous answer failed grounding verification: "
        f"{verdict.summary()}.\n\n"
        "These are the ONLY policy sections retrieved this turn, and the only "
        f"facts you may state:\n{sections}\n\n"
        "Rewrite your answer using only what appears above. If the answer is "
        "not in that text, say you do not have that information and offer a "
        "human agent. Do not state any figure that is not written above."
    )
