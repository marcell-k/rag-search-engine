import argparse

from rag_engine.generation.pipeline import augment


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    sub_parsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    for name, help_text in [
        ("rag", "Perform RAG (search + generate answer)"),
        ("summarize", "Summarize search results"),
        ("citations", "Summarize search results with citations"),
        ("question", "Detailed answer with citations"),
    ]:
        sp = sub_parsers.add_parser(name, help=help_text)
        sp.add_argument("query", type=str, help="Search query for RAG")
        sp.add_argument("--limit", type=int, default=5, help="Number of results")

    args = parser.parse_args()

    match args.command:
        case "rag":
            augment(args.query, args.limit, mode="generation/answer_question")

        case "summarize":
            augment(args.query, args.limit, mode="generation/summarization")

        case "citations":
            augment(args.query, args.limit, mode="generation/answer_with_citations")

        case "question":
            augment(args.query, args.limit, mode="generation/answer_question_detailed")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
