import logging
from typing import TYPE_CHECKING

from google.genai import types

from rag_engine.config import PROJECT_ROOT

if TYPE_CHECKING:
    from google import genai

logger = logging.getLogger(__name__)


class QueryRewriter:
    def __init__(self, client: genai.Client, model: str = "gemma-4-31b-it") -> None:
        self.client = client
        self.model = model

    def _load_prompt(self, mode: str) -> str:
        """Dynamically load the prompt based on the enhancement mode."""
        prompt_path = PROJECT_ROOT / "src" / "rag_engine" / "query_processing" / "prompts" / f"{mode}.md"

        with prompt_path.open("r") as f:
            return f.read().strip()

    def rewrite(self, raw_query: str, mode: str = "spell") -> str:
        """Format the prompt with the query and calls Gemini."""
        try:
            # 1. Load the specific prompt (e.g., "spell.md" or "rewrite.md")
            prompt_template = self._load_prompt(mode)

            # 2. Format the user's query into the {query} placeholder
            formatted_prompt = prompt_template.format(query=raw_query)

            # 3. Call the model using the fully formatted string
            config = types.GenerateContentConfig(temperature=0.0)
            response = self.client.models.generate_content(model=self.model, contents=formatted_prompt, config=config)

            cleaned_query = response.text.strip()
            logger.info(f"Enhanced query ({mode}): '{raw_query}' -> '{cleaned_query}'")
            return cleaned_query

        except Exception as e:
            logger.error(f"Query rewriting failed: {e}. Falling back to raw query.")
            return raw_query
