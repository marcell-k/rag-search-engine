import argparse

from rag_engine.evaluation import runner


def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    results = runner.run(limit=limit)

    print(f"k={limit}\n")
    for r in results:
        print(f"- Query: {r.query}")
        print(f"  - Precision@{limit}: {r.precision:.4f}")
        print(f"  - Recall@{limit}: {r.recall:.4f}")
        print(f"  - F1 Score: {r.f1:.4f}")
        print(f"  - Retrieved: {', '.join(r.retrieved)}")
        print(f"  - Relevant: {', '.join(r.relevant)}")
        print()


if __name__ == "__main__":
    main()
