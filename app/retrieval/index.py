"""Hybrid retrieval over the policy doc: dense + BM25, fused with RRF.

Neither retriever is sufficient alone here. Semantic search handles "can I
send this back?" against "Return window", which BM25 misses entirely for lack
of a shared term. BM25 handles "TR-4521" and "final sale" -- rare exact tokens
a 384-dimension embedding smears into neighbouring clauses. Fusing by rank
rather than score avoids having to calibrate two incomparable scales.
"""

import logging
import re
from dataclasses import dataclass
from functools import lru_cache

import chromadb
from rank_bm25 import BM25Okapi

from app.retrieval.chunker import Chunk, chunk_policy

logger = logging.getLogger(__name__)

# Favours semantic recall, per the brief. Raising it hurts exact-token queries
# like "final sale"; lowering it hurts paraphrases like "send it back".
SEMANTIC_WEIGHT = 0.6
# RRF damping. The usual constant is 60, which comes from TREC-scale runs over
# millions of documents. Over 28 chunks it flattens the curve almost to nothing
# -- rank 1 and rank 12 end up 18% apart -- so fusion starts rewarding chunks
# that are mediocre in BOTH lists over a chunk one retriever ranked first. At 2
# the rank signal survives, which is what a corpus this small needs.
RRF_K = 2


@dataclass
class Hit:
    chunk: Chunk
    score: float

    def as_dict(self) -> dict:
        return {
            "citation": self.chunk.citation,
            "section": self.chunk.section,
            "clause": self.chunk.clause,
            "text": self.chunk.text,
            "score": round(self.score, 5),
        }


# A short stoplist, not an NLP dependency. Without it BM25 scores "how do I
# have to" against every clause in the document and returns document order.
STOPWORDS = frozenset("""
a an and are as at be been by can could do does did for from get got had has
have how i if in into is it its me my of on or our so than that the their them
then there these they this to was were what when where which who will with
would you your im ive dont cant s t
""".split())


def _tokenise(text: str, drop_stopwords: bool = True) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    if not drop_stopwords:
        return tokens
    kept = [t for t in tokens if t not in STOPWORDS]
    # A query of nothing but stopwords keeps its tokens rather than becoming
    # empty, which would silently match everything.
    return kept or tokens


class PolicyIndex:
    def __init__(self, chunks: list[Chunk] | None = None):
        self.chunks = chunks if chunks is not None else chunk_policy()
        self._by_id = {c.chunk_id: c for c in self.chunks}

        # In-process and in-memory: 28 chunks, rebuilt in about a second at
        # startup. A persistent store would be one more thing to invalidate
        # when the policy changes.
        client = chromadb.EphemeralClient()
        self._collection = client.create_collection(
            name="trendly_policy",
            metadata={"hnsw:space": "cosine"},
        )
        self._collection.add(
            ids=[c.chunk_id for c in self.chunks],
            documents=[c.text for c in self.chunks],
        )

        self._bm25 = BM25Okapi([_tokenise(c.text) for c in self.chunks])
        logger.info("policy index built: %s chunks", len(self.chunks))

    def _semantic_ranking(self, query: str, depth: int) -> list[str]:
        result = self._collection.query(query_texts=[query], n_results=depth)
        return result["ids"][0]

    def _keyword_ranking(self, query: str, depth: int) -> list[str]:
        """Ranked chunk ids, excluding anything BM25 did not actually match.

        Zero-scoring chunks must be dropped, not ranked. Keeping them means a
        query with no lexical overlap still hands rank-1 fusion credit to
        whichever chunk happens to sort first -- which is document order, and
        is how "arrived broken" ended up retrieving the shipping preamble.
        """
        scores = self._bm25.get_scores(_tokenise(query))
        ordered = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.chunks[i].chunk_id for i in ordered[:depth] if scores[i] > 0]

    def search(self, query: str, k: int = 4) -> list[Hit]:
        """Top-k chunks by reciprocal rank fusion of both retrievers."""
        if not query.strip():
            return []

        depth = min(len(self.chunks), max(k * 3, 10))
        fused: dict[str, float] = {}

        for rank, chunk_id in enumerate(self._semantic_ranking(query, depth)):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + SEMANTIC_WEIGHT / (RRF_K + rank + 1)
        for rank, chunk_id in enumerate(self._keyword_ranking(query, depth)):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + (1 - SEMANTIC_WEIGHT) / (RRF_K + rank + 1)

        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [Hit(chunk=self._by_id[cid], score=score) for cid, score in ranked]


@lru_cache
def get_index() -> PolicyIndex:
    return PolicyIndex()
