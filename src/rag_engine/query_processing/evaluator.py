import json
import re
from typing import TYPE_CHECKING

from google.genai import types

from rag_engine.config import PROJECT_ROOT

if TYPE_CHECKING:
    from google import genai

    from rag_engine.models import HybridSearchResult

_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class SearchEvaluator:
    def __init__(self, client: genai.Client, model: str = "gemma-4-31b-it") -> None:
        self.client = client
        self.model = model

    def _load_prompt(self) -> str:
        path = PROJECT_ROOT / "src" / "rag_engine" / "query_processing" / "prompts" / "evaluate.md"
        with path.open("r") as f:
            return f.read().strip()

    def _format_results(self, results: list[HybridSearchResult]) -> list[str]:
        return [
            f"{i}. Title: {res['title']}\n   Description: {res['description'][:200]}"
            for i, res in enumerate(results, start=1)
        ]

    def evaluate(self, query: str, results: list[HybridSearchResult]) -> list[tuple[str, int]]:
        template = self._load_prompt()
        formatted = self._format_results(results)

        prompt = template.replace("{chr(10).join(formatted_results)}", "\n".join(formatted))
        prompt = prompt.replace("{query}", query)

        config = types.GenerateContentConfig(temperature=0.0)
        response = self.client.models.generate_content(model=self.model, contents=prompt, config=config)

        if response.text is None:
            raise ValueError("LLM returned no text for evaluation.")

        cleaned = _CODE_FENCE_RE.sub("", response.text).strip()
        scores: list[int] = json.loads(cleaned)
        return [(res["title"], scores[i]) for i, res in enumerate(results)]
