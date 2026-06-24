from typing import TYPE_CHECKING

from rag_engine.keyword.search import bm25_search

if TYPE_CHECKING:
    from rag_engine.index import InvertedIndex
    from rag_engine.models import HybridSearchResult, SearchResult
    from rag_engine.semantic.embedder import ChunkedSemanticSearch


class HybridSearch:
    def __init__(self, inverted_index: InvertedIndex, semantic_search: ChunkedSemanticSearch) -> None:
        """Inject pre-initialized and pre-loaded search backends."""
        self.ii = inverted_index
        self.semantic_search = semantic_search

    def _bm25_search(self, query: str, limit: int) -> list[SearchResult]:
        return bm25_search(self.ii, query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[HybridSearchResult]:
        bm_results = self._bm25_search(query, limit * 500)
        sem_results = self.semantic_search.search_chunks(query, limit * 500)
        combined_results = combine_search_results(bm_results, sem_results, alpha=alpha)
        return combined_results[:limit]

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[HybridSearchResult]:
        bm_results = self._bm25_search(query, limit * 500)
        sem_results = self.semantic_search.search_chunks(query, limit * 500)
        return rrf_combined_results(bm_results, sem_results, k)[:limit]


def rrf_combined_results(
    bm_results: list[SearchResult], sem_results: list[SearchResult], k: int = 60
) -> list[HybridSearchResult]:
    combined: dict[int, HybridSearchResult] = {}

    for i, bm_res in enumerate(bm_results, start=1):
        doc_id = bm_res["doc_id"]
        combined[doc_id] = {
            "doc_id": doc_id,
            "bm_rank": i,
            "bm_score": rrf_score(i, k),
            "sem_score": 0.0,
            "sem_rank": None,
            "hybrid_score": 0.0,
            "hybrid_rank": None,
            "title": bm_res["title"],
            "description": bm_res["description"],
            "score": 0.0,
        }
    for i, sem_res in enumerate(sem_results, start=1):
        doc_id = sem_res["doc_id"]
        if doc_id not in combined:
            combined[doc_id] = {
                "doc_id": doc_id,
                "bm_score": 0.0,
                "bm_rank": None,
                "sem_score": rrf_score(i, k),
                "sem_rank": i,
                "hybrid_rank": None,
                "hybrid_score": 0.0,
                "title": sem_res["title"],
                "description": sem_res["description"],
                "score": 0.0,
            }
        # FIX: Only assign rank/score if it's the first (best) chunk seen for this document
        elif combined[doc_id]["sem_rank"] is None:
            combined[doc_id]["sem_rank"] = i
            combined[doc_id]["sem_score"] = rrf_score(i, k)

    for doc_id in combined:
        hybrid_total = combined[doc_id]["bm_score"] + combined[doc_id]["sem_score"]
        combined[doc_id]["hybrid_score"] = hybrid_total
        combined[doc_id]["score"] = hybrid_total

    results = sorted(combined.values(), key=lambda x: x["hybrid_score"], reverse=True)
    return results


def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)


def hybrid_score(bm25_score: float, sem_score: float, alpha: float = 0.5) -> float:
    return (alpha * bm25_score) + ((1 - alpha) * sem_score)


def normalize_search_results(results: list[SearchResult], floor_zero: bool = False) -> list[dict]:
    scores = [r["score"] for r in results]
    normalized = minmax_normalization(scores, floor_zero=floor_zero)
    return [
        {
            **result,
            "normalized_score": normalized[i],
        }
        for i, result in enumerate(results)
    ]


def combine_search_results(
    bm_results: list[SearchResult], sem_results: list[SearchResult], alpha: float = 0.5
) -> list[HybridSearchResult]:
    # FIX: Use floor_zero=True for BM25 to keep strong sparse keyword scores intact
    bm_norm = normalize_search_results(bm_results, floor_zero=True)
    sm_norm = normalize_search_results(sem_results, floor_zero=False)

    combined: dict[int, HybridSearchResult] = {}

    # Process Keyword (BM25) Results
    for norm in bm_norm:
        doc_id = norm["doc_id"]
        combined[doc_id] = {
            "doc_id": doc_id,
            "score": 0.0,
            "bm_score": norm["normalized_score"],
            "sem_score": 0.0,
            "hybrid_score": 0.0,
            "bm_rank": None,
            "sem_rank": None,
            "hybrid_rank": None,
            "title": norm["title"],
            "description": norm["description"],
        }

    for norm in sm_norm:
        doc_id = norm["doc_id"]
        if doc_id not in combined:
            combined[doc_id] = {
                "doc_id": doc_id,
                "score": 0.0,
                "bm_score": 0.0,
                "sem_score": norm["normalized_score"],
                "hybrid_score": 0.0,
                "bm_rank": None,
                "sem_rank": None,
                "hybrid_rank": None,
                "title": norm["title"],
                "description": norm["description"],
            }
        combined[doc_id]["sem_score"] = max(combined[doc_id]["sem_score"], norm["normalized_score"])

    for k, v in combined.items():
        hybrid = hybrid_score(v["bm_score"], v["sem_score"], alpha=alpha)
        combined[k]["hybrid_score"] = hybrid
        combined[k]["score"] = hybrid

    results = sorted(combined.values(), key=lambda x: x["hybrid_score"], reverse=True)
    return results


def minmax_normalization(vector: list[float], floor_zero: bool = False) -> list[float]:
    if not vector:
        return []
    min_score = 0.0 if floor_zero else min(vector)
    max_score = max(vector)
    if min_score == max_score:
        return [1.0] * len(vector)
    return [(score - min_score) / (max_score - min_score) for score in vector]
