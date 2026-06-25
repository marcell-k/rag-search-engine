import argparse
import logging

from rag_engine.hybrid import pipeline
from rag_engine.hybrid.search import minmax_normalization

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search")
    sub_parser = parser.add_subparsers(dest="command", help="Available commands")

    search = sub_parser.add_parser("search")
    search.add_argument("text", type=str)
    search.add_argument("--limit", default=5, type=int)

    normalize_parser = sub_parser.add_parser("normalize", help="Normalize a list of scores")
    normalize_parser.add_argument("vector", type=float, nargs="*", help="List of scores to normalize")

    weighted_search_parser = sub_parser.add_parser("weighted-search", help="Hybrid search with weighted average")
    weighted_search_parser.add_argument("query", type=str, help="User query to find relevant information")
    weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="Percent of weight for bm25 (keyword)")
    weighted_search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    rrf_search_parser = sub_parser.add_parser("rrf-search", help="Hybrid search using Reciprocal Rank Fusion")
    rrf_search_parser.add_argument("query", type=str, help="User query")
    rrf_search_parser.add_argument("-k", type=int, default=60)
    rrf_search_parser.add_argument("--limit", type=int, default=5)
    rrf_search_parser.add_argument(
        "--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method"
    )
    rrf_search_parser.add_argument(
        "--rerank-method", choices=["individual", "batch", "cross_encoder"], help="Rerank method"
    )
    rrf_search_parser.add_argument("--debug", action="store_true", default=False, help="Enable debug")
    rrf_search_parser.add_argument(
        "--evaluate", action="store_true", default=False, help="LLM-evaluate result relevance (0-3)"
    )

    args = parser.parse_args()
    if getattr(args, "debug", False):
        logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
        logging.getLogger("rag_engine").setLevel(logging.DEBUG)

    match args.command:
        case "weighted-search":
            components = pipeline.build_pipeline()
            results = components.hybrid_search.weighted_search(args.query, args.alpha, args.limit)
            for i, res in enumerate(results, start=1):
                print(f"{i} {res['title']}")
                print(f"  Hybrid Score: {res['hybrid_score']:.4f}")
                print(f"  BM25: {res['bm_score']:.4f}, Semantic: {res['sem_score']:.4f}")
                print(f"  {res['description'][:40]}")

        case "rrf-search":
            components = pipeline.build_pipeline()
            run = pipeline.rrf_search(
                components,
                query=args.query,
                k=args.k,
                limit=args.limit,
                enhance=args.enhance,
                rerank_method=args.rerank_method,
            )

            if run.enhanced_query:
                print(f"Enhanced query ({run.enhance_mode}): '{run.query}' -> '{run.enhanced_query}'\n")

            if run.rerank_method:
                print(f"Re-ranking top {run.pre_rerank_count} results using {run.rerank_method} method...")
                print(f"Reciprocal Rank Fusion Results for '{run.query}' (k={run.k}):\n")

            for i, res in enumerate(run.results, start=1):
                print(f"{i}. {res['title']}")

                if run.rerank_method == "individual" and "rerank_score" in res:
                    print(f"   Re-rank Score: {res['rerank_score']:.3f}/10")
                elif run.rerank_method == "batch":
                    print(f"   Re-rank Rank: {i}")
                elif run.rerank_method == "cross_encoder" and "rerank_score" in res:
                    print(f"   Cross Encoder Score: {res['rerank_score']:.4f}")

                print(f"   RRF Score: {res['hybrid_score']:.3f}")
                print(f"   BM25 Rank: {res['bm_rank']}, Semantic Rank: {res['sem_rank']}")
                print(f"   {res['description'][:97]}...\n")

            if args.evaluate:
                if not components.evaluator:
                    print("\nEvaluation skipped: no GEMINI_API_KEY set.")
                else:
                    scored = components.evaluator.evaluate(run.query, run.results)
                    print("\nEvaluation Report:")
                    for i, (title, score) in enumerate(scored, start=1):
                        print(f"{i}. {title}: {score}/3")

        case "search":
            components = pipeline.build_pipeline()
            print(components.hybrid_search._bm25_search(args.text, args.limit))

        case "normalize":
            results = minmax_normalization(args.vector)
            for res in results:
                print(f"*{res:.4f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
