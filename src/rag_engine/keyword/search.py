from collections import defaultdict
from typing import TYPE_CHECKING

from rag_engine.preprocessing import tokenizer

if TYPE_CHECKING:
    from rag_engine.index import InvertedIndex, Movie


def search_command(ii: InvertedIndex, query: str, n_results: int = 5) -> list[Movie]:
    res: list[Movie] = []
    seen: set[int] = set()

    query_tokens = tokenizer(query)
    for token in query_tokens:
        matching_doc_ids = ii.get_documents(token)
        for doc_id in matching_doc_ids:
            if doc_id not in seen:
                res.append(ii.docmap[doc_id])
                seen.add(doc_id)
            if len(res) == n_results:
                break
    return res


def bm25_search(ii: InvertedIndex, query: str, limit: int) -> list[tuple[float, Movie]]:
    scores: dict[int, float] = defaultdict(float)
    query_tokens = tokenizer(query)
    for token in query_tokens:
        matching_doc_ids = ii.get_documents(token)
        for doc_id in matching_doc_ids:
            scores[doc_id] += ii.bm25(doc_id, token)

    if not scores:
        return []
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, ii.docmap[doc_id]) for doc_id, score in sorted_docs[:limit]]
