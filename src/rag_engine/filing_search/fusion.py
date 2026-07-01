from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_engine.models import HybridSearchResult, SearchResult


def rrf_combined_results(
    fts_results: list[SearchResult], vec_results: list[SearchResult], k: int = 60
) -> list[HybridSearchResult]:
    combined: dict[str, HybridSearchResult] = {}

    for i, res in enumerate(fts_results, start=1):
        cid = res["chunk_id"]
        combined[cid] = {
            "chunk_id": cid,
            "content": res["content"],
            "cik": res["cik"],
            "company_name": res["company_name"],
            "filing_type": res["filing_type"],
            "period_of_report": res["period_of_report"],
            "sec_item": res["sec_item"],
            "sec_title": res["sec_title"],
            "fts_rank": i,
            "fts_score": rrf_score(i, k),
            "vec_score": 0.0,
            "vec_rank": None,
            "hybrid_score": 0.0,
            "hybrid_rank": None,
            "score": 0.0,
        }

    for i, res in enumerate(vec_results, start=1):
        cid = res["chunk_id"]
        if cid not in combined:
            combined[cid] = {
                "chunk_id": cid,
                "content": res["content"],
                "cik": res["cik"],
                "company_name": res["company_name"],
                "filing_type": res["filing_type"],
                "period_of_report": res["period_of_report"],
                "sec_item": res["sec_item"],
                "sec_title": res["sec_title"],
                "fts_score": 0.0,
                "fts_rank": None,
                "vec_rank": i,
                "vec_score": rrf_score(i, k),
                "hybrid_score": 0.0,
                "hybrid_rank": None,
                "score": 0.0,
            }
        elif combined[cid]["vec_rank"] is None:
            combined[cid]["vec_rank"] = i
            combined[cid]["vec_score"] = rrf_score(i, k)

    for cid in combined:
        total = combined[cid]["fts_score"] + combined[cid]["vec_score"]
        combined[cid]["hybrid_score"] = total
        combined[cid]["score"] = total

    results = sorted(combined.values(), key=lambda x: x["hybrid_score"], reverse=True)
    return results


def rrf_score(rank: int, k: int = 60) -> float:
    return 1 / (k + rank)
