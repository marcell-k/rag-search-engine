import json
import logging
import re
import time
from typing import TYPE_CHECKING, TypeVar

from google.genai import types

from rag_engine.config import PROJECT_ROOT

if TYPE_CHECKING:
    from collections.abc import Sequence

    from google import genai

    from rag_engine.models import SearchResult

logger = logging.getLogger(__name__)

_R = TypeVar("_R", bound="SearchResult")

_RANKING_TAG_RE = re.compile(r"<ranking>(.*?)</ranking>", re.DOTALL)
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class SearchReranker:
    def __init__(self, client: genai.Client, model: str = "gemma-4-31b-it") -> None:
        self.client = client
        self.model = model

    def _load_prompt(self, mode: str) -> str:
        prompt_path = PROJECT_ROOT / "src" / "rag_engine" / "query_processing" / "prompts" / f"{mode}.md"
        with prompt_path.open("r") as f:
            return f.read().strip()

    def _call_llm(self, prompt: str) -> str:
        """Shared helper to call the LLM and aggressively clean markdown/whitespace."""
        config = types.GenerateContentConfig(temperature=0.0)
        response = self.client.models.generate_content(model=self.model, contents=prompt, config=config)

        if response.text is None:
            finish_reason = response.candidates[0].finish_reason if response.candidates else None
            block_reason = getattr(response.prompt_feedback, "block_reason", None) if response.prompt_feedback else None
            raise ValueError(f"LLM returned no text (finish_reason={finish_reason}, block_reason={block_reason}).")

        return _CODE_FENCE_RE.sub("", response.text).strip()

    def _rerank_individual(self, query: str, search_results: Sequence[_R], mode: str) -> list[_R]:
        prompt_template = self._load_prompt(mode)
        scored_results: list[tuple[float, _R]] = []

        for i, res in enumerate(search_results):
            formatted_prompt = prompt_template.format(query=query, title=res["title"], description=res["description"])

            cleaned_score = self._call_llm(formatted_prompt)
            res_dict = dict(res)
            res_dict["rerank_score"] = float(cleaned_score)

            scored_results.append((res_dict["rerank_score"], res_dict))  # type: ignore

            if i < len(search_results) - 1:
                logger.info("Waiting 5 seconds before scoring next movie to respect rate limits...")
                time.sleep(5)

        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_results]

    def _rerank_batch(self, query: str, search_results: Sequence[_R], mode: str) -> list[_R]:
        doc_list_str = ""
        for res in search_results:
            doc_list_str += f"\nID: {res['doc_id']}\nTitle: {res['title']}\nDescription: {res['description']}\n"
        prompt_template = self._load_prompt(mode)
        formatted_prompt = prompt_template.format(query=query, doc_list_str=doc_list_str)

        cleaned_text = self._call_llm(formatted_prompt)

        match = _RANKING_TAG_RE.search(cleaned_text)
        if match is None:
            raise ValueError(f"Could not find <ranking> tags in LLM response: {cleaned_text!r}")

        ranked_ids = json.loads(match.group(1).strip())

        res_map = {res["doc_id"]: res for res in search_results}
        ranked_results = []
        for doc_id in ranked_ids:
            if doc_id in res_map:
                ranked_results.append(res_map[doc_id])

        seen_ids = set(ranked_ids)
        for res in search_results:
            if res["doc_id"] not in seen_ids:
                ranked_results.append(res)

        return ranked_results

    def _rerank_cross_encoder(self, query: str, search_results: Sequence[_R]) -> list[_R]:
        from sentence_transformers import CrossEncoder

        cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")

        pairs = [[query, f"{res['title']} - {res['description'][:300]}"] for res in search_results]
        scores = cross_encoder.predict(pairs)

        scored_results = []
        for res, score in zip(search_results, scores, strict=True):
            res_dict = dict(res)
            res_dict["rerank_score"] = float(score)
            scored_results.append(res_dict)

        scored_results.sort(key=lambda doc: doc["rerank_score"], reverse=True)

        return scored_results

    def rerank(self, query: str, search_results: Sequence[_R], mode: str) -> list[_R]:
        """Wrap handle the try/except logic and route to the correct reranking strategy."""
        try:
            if mode == "batch":
                return self._rerank_batch(query, search_results, mode)
            elif mode == "cross_encoder":
                return self._rerank_cross_encoder(query, search_results)
            else:
                return self._rerank_individual(query, search_results, mode)

        except Exception as e:
            logger.error(f"Reranking failed: {e}. Falling back to original results.")
            return list(search_results)
