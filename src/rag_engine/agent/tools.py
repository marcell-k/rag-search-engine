import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from rag_engine.hybrid.pipeline import PipelineComponents
    from rag_engine.models import HybridSearchResult, Movie, SearchResult


@dataclass
class AgentTool:
    name: str
    description: str
    fn: Callable[[str], Sequence[SearchResult]]


def build_tools(
    components: PipelineComponents,
    documents: list[Movie],
) -> dict[str, AgentTool]:
    """Build the tool registry from live pipeline components and the document corpus."""

    def _regex_over_corpus(pattern: str) -> list[SearchResult | HybridSearchResult]:
        """Scan every document's title + description with a regex pattern."""
        results: list[SearchResult] = []
        try:
            compiled = re.compile(pattern, re.IGNORECASE)
        except re.error:
            return results

        for doc in documents:
            if compiled.search(f"{doc['title']} {doc['description']}"):
                results.append(
                    {
                        "doc_id": doc["id"],
                        "score": 1.0,
                        "title": doc["title"],
                        "description": doc["description"],
                    }
                )
        return results[:10]

    def keyword_search(query: str) -> list[SearchResult]:
        from rag_engine.keyword.search import bm25_search

        return bm25_search(components.hybrid_search.ii, query, limit=10)

    def semantic_search(query: str) -> list[SearchResult]:
        return components.hybrid_search.semantic_search.search_chunks(query, limit=10)

    def regex_search(pattern: str) -> list[SearchResult]:
        return _regex_over_corpus(pattern)

    def genre_search(genre: str) -> list[SearchResult]:
        return _regex_over_corpus(genre)

    def actor_search(actor: str) -> list[SearchResult]:
        return _regex_over_corpus(actor)

    def hybrid_rrf_search(query: str) -> list[HybridSearchResult]:
        return components.hybrid_search.rrf_search(query, k=60, limit=10)

    return {
        "hybrid_search": AgentTool(
            name="hybrid_search",
            description=(
                "Hybrid BM25 + semantic search fused with RRF. Best all-round tool — "
                "use this first; fall back to narrower tools only if results are weak."
            ),
            fn=hybrid_rrf_search,
        ),
        "keyword_search": AgentTool(
            name="keyword_search",
            description=("BM25 keyword search. Best for specific terms, movie titles, or character names."),
            fn=keyword_search,
        ),
        "semantic_search": AgentTool(
            name="semantic_search",
            description=(
                "Embedding-based semantic search. Best for themes, moods, "
                "and abstract concepts like 'survival' or 'redemption'."
            ),
            fn=semantic_search,
        ),
        "regex_search": AgentTool(
            name="regex_search",
            description=(
                "Regex pattern match across title and description. Best for "
                "specific phrases like 'bear attack' or 'wilderness survival'."
            ),
            fn=regex_search,
        ),
        "genre_search": AgentTool(
            name="genre_search",
            description=(
                "Filter movies by genre keyword: horror, thriller, comedy, adventure, drama, sci-fi, romance, etc."
            ),
            fn=genre_search,
        ),
        "actor_search": AgentTool(
            name="actor_search",
            description="Find movies that mention a specific actor by name.",
            fn=actor_search,
        ),
    }
