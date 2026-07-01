import argparse
import asyncio

from db.client import DatabaseClient
from db.repository import ChunkRepository
from rag_engine.embedding.semantic import SemanticChunk
from rag_engine.filing_search.pipeline import search_hybrid


async def _run_fts(query: str, limit: int, cik: str | None) -> None:
    client = DatabaseClient()
    await client.connect()
    try:
        repo = ChunkRepository(client)
        results = await repo.search_fts(query, limit, cik)
        for i, r in enumerate(results, start=1):
            print(f"{i}. [{r['chunk_id']}] {r['company_name']} ({r['filing_type']}) score={r['score']:.4f}")
            print(f"   {r['content'][:220]}...")
    finally:
        await client.disconnect()


async def _run_hybrid(query: str, limit: int, cik: str | None) -> None:
    client = DatabaseClient()
    await client.connect()
    try:
        repo = ChunkRepository(client)
        sc = SemanticChunk()
        results = await search_hybrid(query, cik, sc, repo, limit=limit)
        for i, r in enumerate(results, start=1):
            print(f"{i}. [{r['chunk_id']}] {r['company_name']} ({r['filing_type']}) hybrid={r['hybrid_score']:.4f}")
            print(f"   {r['content'][:220]}...")
    finally:
        await client.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Filing Search CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    fts = sub.add_parser("fts", help="Full-text search")
    fts.add_argument("query", type=str)
    fts.add_argument("--limit", type=int, default=5)
    fts.add_argument("--cik", type=str, default=None)

    hybrid = sub.add_parser("hybrid", help="Keyword + semantic search")
    hybrid.add_argument("query", type=str)
    hybrid.add_argument("--limit", type=int, default=5)
    hybrid.add_argument("--cik", type=str, default=None)

    args = parser.parse_args()

    match args.command:
        case "fts":
            asyncio.run(_run_fts(args.query, args.limit, args.cik))
        case "hybrid":
            asyncio.run(_run_hybrid(args.query, args.limit, args.cik))
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
