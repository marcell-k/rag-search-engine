import argparse
import math

from rag_engine.index import InvertedIndex
from rag_engine.keyword.search import bm25_search, search_command
from rag_engine.preprocessing import tokenize_first_term


def build_command() -> None:
    ii = InvertedIndex()
    ii.build()
    ii.save()


def tf_command(ii: InvertedIndex, doc_id: int, term: str) -> int:
    token = tokenize_first_term(term)
    return ii.get_tf(doc_id, token)


def idf_command(ii: InvertedIndex, term: str) -> float:
    token = tokenize_first_term(term)
    term_match_doc_count = len(ii.get_documents(token))
    total_doc_count = len(ii.docmap)
    idf_value = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
    return idf_value


def tfidf_command(ii: InvertedIndex, doc_id: int, term: str) -> float:
    idf_value = idf_command(ii, term)
    tf_value = tf_command(ii, doc_id, term)
    return idf_value * tf_value


def bm25_idf_command(ii: InvertedIndex, term: str) -> float:
    token = tokenize_first_term(term)
    return ii.get_bm25_idf(token)


def bm25_tf_command(ii: InvertedIndex, doc_id: int, term: str) -> float:
    token = tokenize_first_term(term)
    return ii.get_bm25_tf(doc_id, token)


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    _ = subparsers.add_parser("build", help="Build cache")

    tf_parser = subparsers.add_parser("tf", help="term frequency")
    tf_parser.add_argument("doc_id", type=int, help="The document ID")
    tf_parser.add_argument("term", type=str, help="The term to check")

    idf_parser = subparsers.add_parser("idf", help="inverse document frequency (IDF)")
    idf_parser.add_argument("term", type=str, help="The term to check")

    tfidf_parser = subparsers.add_parser("tfidf", help="TF-IDF")
    tfidf_parser.add_argument("doc_id", type=int, help="The document ID")
    tfidf_parser.add_argument("term", type=str, help="The term to check")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="TF-IDF")
    bm25_idf_parser.add_argument("term", type=str, help="The term to check")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("-limit", type=int, default=5)

    _ = search_parser.add_argument("query", type=str, help="Search query")
    args = parser.parse_args()

    ii = InvertedIndex()
    if args.command != "build":
        ii.load()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search_command(ii, args.query)
            for i, res in enumerate(results, start=1):
                print(f"{i}. {res['title']}")

        case "build":
            build_command()

        case "tf":
            tf = tf_command(ii, args.doc_id, args.term)
            print(tf)

        case "idf":
            idf = idf_command(ii, args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            tf_idf = tfidf_command(ii, args.doc_id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

        case "bm25idf":
            bm25idf = bm25_idf_command(ii, args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")

        case "bm25tf":
            bm25tf = bm25_tf_command(ii, args.doc_id, args.term)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")

        case "bm25search":
            res = bm25_search(ii, args.query, args.limit)
            for i, r in enumerate(res, start=1):
                print(f"{i}. ({r['doc_id']}) {r['title']} - Score: {r['score']:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
