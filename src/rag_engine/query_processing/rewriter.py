import logging
from typing import TYPE_CHECKING

from google.genai import types

from rag_engine.config import PROJECT_ROOT

if TYPE_CHECKING:
    from google import genai

logger = logging.getLogger(__name__)


class QueryRewriter:
    def __init__(self, client: genai.Client, model: str = "gemma-4-26b-a4b-it") -> None:
        """Initialize the rewriter and dynamically loads the prompt from the .md file."""
        self.client = client
        self.model = model
        self.system_instruction = self._load_prompt()

    def _load_prompt(self) -> str:
        prompt_path = PROJECT_ROOT / "src" / "rag_engine" / "query_processing" / "prompts" / "spelling.md"
        with prompt_path.open("r") as f:
            return f.read().strip()

    def rewrite(self, raw_query: str) -> str:
        """Fix spelling and optimizes the query string using the loaded markdown instructions."""
        try:
            config = types.GenerateContentConfig(
                system_instruction=self.system_instruction,
                temperature=0.0,
            )

            response = self.client.models.generate_content(model=self.model, contents=raw_query, config=config)

            cleaned_query = response.text.strip()
            logger.info(f"Rewrote query: '{raw_query}' -> '{cleaned_query}'")
            return cleaned_query

        except Exception as e:
            logger.error(f"Query rewriting failed: {e}. Falling back to raw query.")
            return raw_query
