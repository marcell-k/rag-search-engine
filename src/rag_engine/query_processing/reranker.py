import json
import logging
import re
import time
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rag_engine.llm import LLM
    from rag_engine.models import SearchResult

logger = logging.getLogger(__name__)

_R = TypeVar("_R", bound="SearchResult")

_RANKING_TAG_RE = re.compile(r"<ranking>(.*?)</ranking>", re.DOTALL)


class SearchReranker:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def _rerank_individual(self, query: str, search_results: Sequence[_R], mode: str) -> list[_R]:
        prompt_template = self.llm.load_prompt(f"reranking/{mode}")
        scored_results: list[tuple[float, _R]] = []

        for i, res in enumerate(search_results):
            formatted_prompt = prompt_template.format(query=query, title=res["sec_title"], description=res["content"])

            cleaned_score = self.llm.generate(formatted_prompt)
            res_dict = dict(res)
            res_dict["rerank_score"] = float(cleaned_score)

            scored_results.append((res_dict["rerank_score"], res_dict))  # type: ignore

            if i < len(search_results) - 1:
                logger.info("Waiting 5 seconds before scoring next chunk to respect rate limits...")
                time.sleep(5)

        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_results]

    def _rerank_batch(self, query: str, search_results: Sequence[_R], mode: str) -> list[_R]:
        doc_list_str = ""
        for res in search_results:
            doc_list_str += f"\nID: {res['chunk_id']}\nTitle: {res['sec_title']}\nContent: {res['content']}\n"
        prompt_template = self.llm.load_prompt("reranking/" + mode)
        formatted_prompt = prompt_template.format(query=query, doc_list_str=doc_list_str)

        cleaned_text = self.llm.generate(formatted_prompt)

        match = _RANKING_TAG_RE.search(cleaned_text)
        if match is None:
            raise ValueError(f"Could not find <ranking> tags in LLM response: {cleaned_text!r}")

        ranked_ids = json.loads(match.group(1).strip())

        res_map = {res["chunk_id"]: res for res in search_results}
        ranked_results = []
        for chunk_id in ranked_ids:
            if chunk_id in res_map:
                ranked_results.append(res_map[chunk_id])

        seen_ids = set(ranked_ids)
        for res in search_results:
            if res["chunk_id"] not in seen_ids:
                ranked_results.append(res)

        return ranked_results

    def _rerank_cross_encoder(self, query: str, search_results: Sequence[_R]) -> list[_R]:
        from sentence_transformers import CrossEncoder

        cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")

        pairs = [[query, f"{res['sec_title']} - {res['content'][:300]}"] for res in search_results]
        scores = cross_encoder.predict(pairs)

        scored_results = []
        for res, score in zip(search_results, scores, strict=True):
            res_dict = dict(res)
            res_dict["rerank_score"] = float(score)
            scored_results.append(res_dict)

        scored_results.sort(key=lambda doc: doc["rerank_score"], reverse=True)

        return scored_results

    def rerank(self, query: str, search_results: Sequence[_R], mode: str) -> list[_R]:
        """Wrap try/except, route to correct reranking strategy."""
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
