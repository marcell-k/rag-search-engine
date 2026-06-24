import argparse

from rag_engine.data_loader import load_data
from rag_engine.semantic.embedder import ChunkedSemanticSearch, SemanticSearch, verify_embeddings, verify_model
from rag_engine.semantic.search import (
    embed_query,
    embed_text,
    fixed_size_chunking,
    search,
    search_chunked,
    semantic_chunking,
)


def verify_model_command(ss: SemanticSearch) -> None:
    model, max_length = verify_model(ss)
    print(f"Model loaded: {model}")
    print(f"Max sequence length: {max_length}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    _ = subparsers.add_parser("verify", help="Verify the embedding model loads properly")
    _ = subparsers.add_parser("verify-embeddings", help="Verify embedding documents")

    embed_parser = subparsers.add_parser("embed-text", help="Encode text with embedding model")
    embed_parser.add_argument("text", type=str, help="Text to be encdoded")

    embed_query_parser = subparsers.add_parser("embed-query", help="Embed a search query")
    embed_query_parser.add_argument("query", type=str, help="The query string to embed")

    search_parser = subparsers.add_parser("search", help="Semantic search")
    search_parser.add_argument("query", type=str)
    search_parser.add_argument("-limit", default=5)

    chunk_parser = subparsers.add_parser("chunk")
    chunk_parser.add_argument("text", type=str)
    chunk_parser.add_argument("--chunk_size", type=int, default="200")
    chunk_parser.add_argument("--overlap", type=float, default=0.2)

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk")
    semantic_chunk_parser.add_argument("text", type=str)
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4)
    semantic_chunk_parser.add_argument("--overlap", type=float, default=0)

    _ = subparsers.add_parser("embed-chunks", help="Create embeddings for document chunks")

    search_chunked_parser = subparsers.add_parser("search-chunked")
    search_chunked_parser.add_argument("query", type=str)
    search_chunked_parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    ss = SemanticSearch()
    match args.command:
        case "verify":
            verify_model_command(ss)

        case "embed-text":
            embed_text(ss, args.text)

        case "verify-embeddings":
            verify_embeddings(ss)

        case "embed-query":
            embed_query(args.query)

        case "search":
            search(ss, args.query, args.limit)
        case "chunk":
            chunks = fixed_size_chunking(args.text, args.chunk_size, args.overlap)
            print(f"Chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks, start=1):
                print(f"{i}. {chunk}")

        case "semantic_chunk":
            chunks = semantic_chunking(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.text)}")
            for i, chunk in enumerate(chunks, start=1):
                print(f"{i}. {chunk}")

        case "embed-chunks":
            movies = load_data()
            css = ChunkedSemanticSearch()
            embeddings = css.load_or_create_chunk_embeddings(movies)
            assert embeddings is not None
            print(f"Generated {len(embeddings)} chunked embeddings")

        case "search-chunked":
            search_chunked(args.query, args.limit)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
