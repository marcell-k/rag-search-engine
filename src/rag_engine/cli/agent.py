import argparse
import asyncio
import logging
import os

from dotenv import load_dotenv
from google import genai

from db.client import DatabaseClient
from db.repository import ChunkRepository
from rag_engine.agent.loop import run_agent
from rag_engine.agent.tools import build_tools
from rag_engine.embedding.semantic import SemanticChunk
from rag_engine.llm import LLM


async def _main(args: argparse.Namespace) -> None:
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: no LLM configured. Set GEMINI_API_KEY and retry.")
        return
    client = genai.Client(api_key=api_key)
    llm = LLM(client)

    db_client = DatabaseClient()
    await db_client.connect()
    try:
        cr = ChunkRepository(db_client)
        sc = SemanticChunk()
        tools = build_tools(cr, sc)

        print(f'\nAgentic search: "{args.query}"')
        print("─" * 60)

        run = await run_agent(
            query=args.query,
            tools=tools,
            llm=llm,
            limit=args.limit,
            max_iterations=args.max_iterations,
        )

        print(f"\nSearch trace ({len(run.iterations)} step(s)):")
        for record in run.iterations:
            print(f"  [{record.step}] {record.tool}('{record.query}')")
            print(f"       → {record.found} hit(s), {record.new_unique} new unique | {record.reasoning}")

        print(f"\nTop {len(run.results)} result(s):")
        for i, res in enumerate(run.results, start=1):
            print(f"  {i}. {res['sec_title']} — {res['company_name']} ({res['filing_type']})")

        print(f"\nAnswer:\n{run.answer}\n")
    finally:
        await db_client.disconnect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Agentic RAG over SEC filings.")
    parser.add_argument("query", type=str)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-iterations", type=int, default=6)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
        logging.getLogger("rag_engine").setLevel(logging.DEBUG)

    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
