from dataclasses import dataclass

from rag_engine.data_loader import load_golden
from rag_engine.evaluation import metrics
from rag_engine.hybrid import pipeline


@dataclass
class EvaluationResult:
    """Precision/recall/F1 for a single golden test case, ready for the CLI to print."""

    query: str
    precision: float
    recall: float
    f1: float
    retrieved: list[str]
    relevant: list[str]


def run(limit: int, k: int = 60) -> list[EvaluationResult]:
    """Run every golden test case through the RRF pipeline and score it."""
    test_cases = load_golden()
    components = pipeline.build_pipeline()

    results: list[EvaluationResult] = []
    for test_case in test_cases:
        query: str = test_case["query"]
        relevant_titles: list[str] = test_case["relevant_docs"]

        search_run = pipeline.rrf_search(components, query=query, k=k, limit=limit)
        retrieved_titles = [r["title"] for r in search_run.results]

        results.append(
            EvaluationResult(
                query=query,
                precision=metrics.precision(retrieved_titles, relevant_titles),
                recall=metrics.recall(retrieved_titles, relevant_titles),
                f1=metrics.f1_score(retrieved_titles, relevant_titles),
                retrieved=retrieved_titles,
                relevant=relevant_titles,
            )
        )

    return results
