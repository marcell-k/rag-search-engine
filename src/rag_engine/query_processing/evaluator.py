import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_engine.llm import LLM
    from rag_engine.models import SearchResult


class SearchEvaluator:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def _format_results(self, results: list[SearchResult]) -> list[str]:
        return [
            f"{i}. {res['sec_title']}\n   Content: {res['content'][:200]}" for i, res in enumerate(results, start=1)
        ]

    def evaluate(self, query: str, results: list[SearchResult]) -> list[tuple[str, int]]:
        template = self.llm.load_prompt("evaluation/evaluate")
        formatted = self._format_results(results)

        prompt = template.format(query=query, results="\n".join(formatted))

        cleaned = self.llm.generate(prompt)
        scores: list[int] = json.loads(cleaned)
        return [(res["sec_title"], scores[i]) for i, res in enumerate(results)]
