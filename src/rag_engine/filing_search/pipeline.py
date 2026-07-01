import asyncio
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from google import genai

from db.client import DatabaseClient
from db.repository import ChunkRepository
from rag_engine.embedding.semantic import SemanticChunk
from rag_engine.filing_search.fusion import rrf_combined_results
from rag_engine.llm import LLM
from rag_engine.query_processing.evaluator import SearchEvaluator
from rag_engine.query_processing.reranker import SearchReranker
from rag_engine.query_processing.rewriter import QueryRewriter

if TYPE_CHECKING:
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


@dataclass
class FilingPipelineComponents:
    """Live DB client, embedder, repo, LLM helpers — one build, reused per query."""

    client: DatabaseClient
    repo: ChunkRepository
    embedder: SemanticChunk
    llm: LLM | None
    rewriter: QueryRewriter | None
    reranker: SearchReranker | None
    evaluator: SearchEvaluator | None


async def build_pipeline() -> FilingPipelineComponents:
    client = DatabaseClient()
    await client.connect()
    repo = ChunkRepository(client)
    sc = SemanticChunk()

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    genai_client = genai.Client(api_key=api_key) if api_key else None

    llm = LLM(genai_client) if genai_client else None
    rewriter = QueryRewriter(llm) if llm else None
    reranker = SearchReranker(llm) if llm else None
    evaluator = SearchEvaluator(llm) if llm else None

    return FilingPipelineComponents(
        client=client,
        repo=repo,
        embedder=sc,
        llm=llm,
        rewriter=rewriter,
        reranker=reranker,
        evaluator=evaluator,
    )
