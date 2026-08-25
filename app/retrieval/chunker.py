"""Heading-aware chunking of trendly_policy.md.

The document does not use h3s. Its real units are bold run-in clauses --
`**2.3 Non-returnable categories.** ...` -- sitting under `## 2. Returns`.
Splitting on markdown headings alone would produce a seven-clause Returns
chunk, and a citation of "Returns" is not specific enough to check an answer
against. So sections are split again at clause boundaries, and each chunk
keeps its section so it can cite "Returns -> 2.3 Non-returnable categories".

Splitting only AT clause starts also keeps a clause's bullet list or table
attached to it: the 3.1 refund table and the 2.3 exclusion list travel with
the sentence that introduces them.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings

SECTION_RE = re.compile(r"^##\s+(.*)$", re.MULTILINE)
# A clause opens a line with its number and title inside a bold run.
CLAUSE_RE = re.compile(r"^\*\*(\d+\.\d+)\s+(.+?)\.\*\*", re.MULTILINE)
RULE_RE = re.compile(r"^-{3,}\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    section: str          # "2. Returns"
    clause: str | None    # "2.3 Non-returnable categories"
    text: str

    @property
    def citation(self) -> str:
        """What the agent quotes, and what the citation guard checks against."""
        section = re.sub(r"^\d+\.\s*", "", self.section)
        if not self.clause:
            return section
        return f"{section} -> {self.clause}"


def _clean(text: str) -> str:
    return RULE_RE.sub("", text).strip()


def chunk_policy(path: Path | None = None) -> list[Chunk]:
    raw = (path or get_settings().policy_path).read_text(encoding="utf-8")
    chunks: list[Chunk] = []

    matches = list(SECTION_RE.finditer(raw))
    preamble = _clean(raw[: matches[0].start()] if matches else raw)
    if preamble:
        chunks.append(
            Chunk(chunk_id="preamble", section="Preamble", clause=None, text=preamble)
        )

    for index, match in enumerate(matches):
        section = match.group(1).strip()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        body = raw[match.end() : end]

        clauses = list(CLAUSE_RE.finditer(body))
        if not clauses:
            # Sections like "7. What the assistant must not do" are a bare list.
            text = _clean(body)
            if text:
                chunks.append(
                    Chunk(
                        chunk_id=f"s{index}",
                        section=section,
                        clause=None,
                        text=f"{section}\n\n{text}",
                    )
                )
            continue

        lead = _clean(body[: clauses[0].start()])
        if lead:
            chunks.append(
                Chunk(
                    chunk_id=f"s{index}-lead",
                    section=section,
                    clause=None,
                    text=f"{section}\n\n{lead}",
                )
            )

        for position, clause in enumerate(clauses):
            stop = (
                clauses[position + 1].start()
                if position + 1 < len(clauses)
                else len(body)
            )
            label = f"{clause.group(1)} {clause.group(2).strip()}"
            text = _clean(body[clause.start() : stop])
            chunks.append(
                Chunk(
                    chunk_id=f"s{index}-c{position}",
                    section=section,
                    clause=label,
                    # The heading rides along so retrieval and the model both
                    # see which section a fact came from.
                    text=f"{section} -> {label}\n\n{text}",
                )
            )

    return chunks
