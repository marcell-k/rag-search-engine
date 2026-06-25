import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_engine.llm import LLM
    from rag_engine.models import HybridSearchResult


class SearchEvaluator:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def _format_results(self, results: list[HybridSearchResult]) -> list[str]:
        return [
            f"{i}. Title: {res['title']}\n   Description: {res['description'][:200]}"
            for i, res in enumerate(results, start=1)
        ]

    def evaluate(self, query: str, results: list[HybridSearchResult]) -> list[tuple[str, int]]:
        template = self.llm.load_prompt("evaluation/evaluate")
        formatted = self._format_results(results)

        prompt = template.format(query=query, results="\n".join(formatted))
        prompt = prompt.replace("{query}", query)

        cleaned = self.llm.generate(prompt)
        scores: list[int] = json.loads(cleaned)
        return [(res["title"], scores[i]) for i, res in enumerate(results)]
