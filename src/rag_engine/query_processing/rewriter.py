import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_engine.llm import LLM

logger = logging.getLogger(__name__)


class QueryRewriter:
    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def rewrite(self, raw_query: str, mode: str = "spell") -> str:
        """Format the prompt with the query and calls Gemini."""
        try:
            prompt = self.llm.load_prompt(mode)
            formatted_prompt = prompt.format(query=raw_query)
            cleaned_query = self.llm.generate(formatted_prompt)

            logger.info(f"Enhanced query ({mode}): '{raw_query}' -> '{cleaned_query}'")
            return cleaned_query

        except Exception as e:
            logger.error(f"Query rewriting failed: {e}. Falling back to raw query.")
            return raw_query
