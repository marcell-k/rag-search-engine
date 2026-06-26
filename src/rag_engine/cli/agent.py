import argparse
import logging

from rag_engine.agent.loop import run_agent
from rag_engine.agent.tools import build_tools
from rag_engine.data_loader import load_data
from rag_engine.hybrid.pipeline import build_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agentic RAG — the LLM chooses which search tools to use and in what order.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  agent "scary bear movies set in a forest"
  agent "films starring Hugh Jackman about survival" --limit 8
  agent "romantic comedies in Paris" --max-iterations 4
  agent "horror movies with twist endings" --debug
        """,
    )
    parser.add_argument("query", type=str, help="Natural-language search query")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        metavar="N",
        help="Maximum number of results to include in the final answer (default: 5)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=6,
        metavar="N",
        help="Maximum agent loop iterations before forcing a stop (default: 6)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print internal tool-picker reasoning and LLM responses",
    )
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
        logging.getLogger("rag_engine").setLevel(logging.DEBUG)

    components = build_pipeline()

    if not components.llm:
        print("Error: no LLM configured. Set GEMINI_API_KEY (or equivalent) and retry.")
        return

    documents = load_data()
    tools = build_tools(components, documents)

    print(f'\nAgentic search: "{args.query}"')
    print("─" * 60)

    run = run_agent(
        query=args.query,
        tools=tools,
        llm=components.llm,
        limit=args.limit,
        max_iterations=args.max_iterations,
    )

    # ── Search trace ──────────────────────────────────────────────── #
    print(f"\nSearch trace ({len(run.iterations)} step(s)):")
    for record in run.iterations:
        print(f"  [{record.step}] {record.tool}('{record.query}')")
        print(f"       → {record.found} hit(s), {record.new_unique} new unique | {record.reasoning}")

    # ── Top results ───────────────────────────────────────────────── #
    print(f"\nTop {len(run.results)} result(s):")
    for i, res in enumerate(run.results, start=1):
        print(f"  {i}. {res['title']}")

    # ── Final answer ──────────────────────────────────────────────── #
    print(f"\nAnswer:\n{run.answer}\n")


if __name__ == "__main__":
    main()
