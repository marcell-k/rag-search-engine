from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence

    from db.repository import ChunkRepository
    from rag_engine.embedding.semantic import SemanticChunk
    from rag_engine.models import HybridSearchResult, SearchResult


@dataclass
class AgentTool:
    name: str
    description: str
    fn: Callable[[str], Awaitable[Sequence[SearchResult]]]


def build_tools(cr: ChunkRepository, sc: SemanticChunk) -> dict[str, AgentTool]:
    """Build tool registry from live repo + embedder."""

    async def fts_search(query: str) -> list[SearchResult]:
        return await cr.search_fts(query, limit=10)

    async def vector_search(query: str) -> list[SearchResult]:
        embedding = sc.generate_embedding(query)
        return await cr.search_vector(embedding, limit=10)

    async def hybrid_search(query: str) -> list[HybridSearchResult]:
        from rag_engine.filing_search.pipeline import search_hybrid

        return await search_hybrid(query, cik=None, sc=sc, cr=cr, limit=10)

    async def topic_search(topic: str) -> list[SearchResult]:
        return await cr.search_by_topic(topic, limit=10)

    async def item_search(item: str) -> list[SearchResult]:
        return await cr.search_by_item(item, limit=10)

    async def filer_search(cik_or_ticker: str) -> list[SearchResult]:
        return await cr.search_by_filer(cik_or_ticker, limit=10)

    return {
        "hybrid_search": AgentTool(
            name="hybrid_search",
            description=(
                "FTS + vector search fused with RRF. Best all-round tool — "
                "use this first; fall back to narrower tools only if results are weak."
            ),
            fn=hybrid_search,
        ),
        "search_fts": AgentTool(
            name="search_fts",
            description="Full-text search. Best for exact terms, defined phrases, or specific figures.",
            fn=fts_search,
        ),
        "search_vector": AgentTool(
            name="search_vector",
            description="Embedding-based semantic search.",
            fn=vector_search,
        ),
        "topic_search": AgentTool(
            name="topic_search",
            description=(
                "Filter chunks by topic tag: revenue_recognition, leases, debt, income_taxes, "
                "risk_factors, cybersecurity, legal_proceedings, etc."
            ),
            fn=topic_search,
        ),
        "item_search": AgentTool(
            name="item_search",
            description="Filter chunks by SEC item number (e.g. 'ITEM 1A', 'ITEM 7', 'ITEM 9A').",
            fn=item_search,
        ),
        "filer_search": AgentTool(
            name="filer_search",
            description="Find chunks by company CIK or ticker.",
            fn=filer_search,
        ),
    }
