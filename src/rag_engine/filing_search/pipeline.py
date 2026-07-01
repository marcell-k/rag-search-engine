import asyncio
from typing import TYPE_CHECKING

from rag_engine.filing_search.fusion import rrf_combined_results

if TYPE_CHECKING:
    from db.repository import ChunkRepository
    from rag_engine.embedding.semantic import SemanticChunk
    from rag_engine.models import HybridSearchResult


async def search_hybrid(
    query: str, cik: str | None, sc: SemanticChunk, cr: ChunkRepository, k: int = 60, limit: int = 10
) -> list[HybridSearchResult]:
    query_embedding = sc.generate_embedding(query)
    fts_results, vec_results = await asyncio.gather(
        cr.search_fts(query, limit * 5, cik), cr.search_vector(query_embedding, limit * 5, cik)
    )
    results = rrf_combined_results(fts_results, vec_results, k)
    return results[:limit]
