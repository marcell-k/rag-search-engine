import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from google import genai

from rag_engine.data_loader import load_data
from rag_engine.hybrid.search import HybridSearch
from rag_engine.index import InvertedIndex
from rag_engine.llm import LLM
from rag_engine.query_processing.evaluator import SearchEvaluator
from rag_engine.query_processing.reranker import SearchReranker
from rag_engine.query_processing.rewriter import QueryRewriter
from rag_engine.semantic.embedder import ChunkedSemanticSearch

if TYPE_CHECKING:
    from rag_engine.models import HybridSearchResult
logger = logging.getLogger(__name__)


@dataclass
class PipelineComponents:
    """Pre-built search backends and LLM helpers, ready for repeated queries."""

    hybrid_search: HybridSearch
    llm: LLM | None
    rewriter: QueryRewriter | None
    reranker: SearchReranker | None
    evaluator: SearchEvaluator | None


@dataclass
class RRFSearchRun:
    """Structured outcome of one end-to-end RRF search, ready for the CLI to print."""

    query: str
    enhanced_query: str | None
    enhance_mode: str | None
    rerank_method: str | None
    k: int
    limit: int
    pre_rerank_count: int | None
    results: list[HybridSearchResult]


def build_pipeline() -> PipelineComponents:
    """Build the index, semantic embeddings, hybrid search, rewriter and reranker."""
    documents = load_data()

    ii = InvertedIndex()
    if not ii._index_file.exists():
        ii.build()
        ii.save()
    else:
        ii.load()

    css = ChunkedSemanticSearch()
    css.load_or_create_chunk_embeddings(documents)

    hybrid_search = HybridSearch(inverted_index=ii, semantic_search=css)

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key) if api_key else None

    llm = LLM(client) if client else None
    rewriter = QueryRewriter(llm) if llm else None
    reranker = SearchReranker(llm) if llm else None
    evaluator = SearchEvaluator(llm) if llm else None

    return PipelineComponents(
        hybrid_search=hybrid_search,
        llm=llm,
        rewriter=rewriter,
        reranker=reranker,
        evaluator=evaluator,
    )


def rrf_search(
    components: PipelineComponents,
    query: str,
    k: int = 60,
    limit: int = 5,
    enhance: str | None = None,
    rerank_method: str | None = None,
) -> RRFSearchRun:
    """Run the full RRF pipeline: optional query enhancement, RRF fusion, optional reranking."""
    logger.debug("Original query: %r", query)

    search_query = query
    enhanced_query: str | None = None
    if enhance in ("spell", "rewrite", "expand") and components.rewriter and components.llm:
        search_query = components.rewriter.rewrite(query, mode="rewriting/" + enhance)
        if search_query != query:
            enhanced_query = search_query
    logger.debug("Query after enhancement: %r", search_query)

    initial_limit = limit * 5 if rerank_method else limit
    results = components.hybrid_search.rrf_search(search_query, k, initial_limit)

    pre_rerank_count: int | None = None
    if rerank_method and components.reranker:
        pre_rerank_count = len(results)
        results = components.reranker.rerank(search_query, results, mode=rerank_method)
        results = results[:limit]

    logger.debug("Final results after re-ranking: %s", [r["title"] for r in results[:limit]])

    return RRFSearchRun(
        query=query,
        enhanced_query=enhanced_query,
        enhance_mode=enhance,
        rerank_method=rerank_method,
        k=k,
        limit=limit,
        pre_rerank_count=pre_rerank_count,
        results=results[:limit],
    )
