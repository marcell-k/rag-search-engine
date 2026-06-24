import argparse
import os

from dotenv import load_dotenv
from google import genai

from rag_engine.data_loader import load_data
from rag_engine.hybrid.search import HybridSearch, minmax_normalization
from rag_engine.index import InvertedIndex
from rag_engine.query_processing.rewriter import QueryRewriter
from rag_engine.semantic.embedder import ChunkedSemanticSearch


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

    args = parser.parse_args()

    documents = load_data()

    ii = InvertedIndex()
    if not ii._index_file.exists():
        ii.build()
        ii.save()
    css = ChunkedSemanticSearch()
    css.load_or_create_chunk_embeddings(documents)
    hs = HybridSearch(inverted_index=ii, semantic_search=css)

    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    rewriter = None
    if api_key:
        client = genai.Client(api_key=api_key)
        rewriter = QueryRewriter(client=client)

    match args.command:
        case "weighted-search":
            results = hs.weighted_search(args.query, args.alpha, args.limit)
            for i, res in enumerate(results, start=1):
                print(f"{i} {res['title']}")
                print(f"  Hybrid Score: {res['hybrid_score']:.4f}")
                print(f"  BM25: {res['bm_score']:.4f}, Semantic: {res['sem_score']:.4f}")
                print(f"  {res['description'][:40]}")

        case "rrf-search":
            search_query = args.query
            if args.enhance in ["spell", "rewrite", "expand"] and rewriter:
                search_query = rewriter.rewrite(args.query, mode=args.enhance)
                if search_query != args.query:
                    print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{search_query}'\n")

            results = hs.rrf_search(search_query, args.k, args.limit)

            for i, res in enumerate(results, start=1):
                print(f"{i} {res['title']}")
                print(f"  RRF Score: {res['hybrid_score']:.4f}")
                print(f"  BM25 Rank: {res['bm_rank']}, Semantic Rank: {res['sem_rank']}")
                print(f"  {res['description'][:30]}")

        case "search":
            documents = load_data()
            print(hs._bm25_search(args.text, args.limit))

        case "normalize":
            results = minmax_normalization(args.vector)
            for res in results:
                print(f"*{res:.4f}")

        case _:
            parser.print_help()
