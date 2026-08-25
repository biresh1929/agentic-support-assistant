"""search_policy -- a read tool over trendly_policy.md.

Returns chunks with their section headings so the agent can cite
"Returns -> 2.3 Non-returnable categories" rather than a chunk number, and so
the citation guard has something concrete to check an answer against.
"""

from app.retrieval.index import get_index

DEFAULT_K = 4


def search_policy(query: str, k: int = DEFAULT_K) -> list[dict]:
    return [hit.as_dict() for hit in get_index().search(query, k=k)]
