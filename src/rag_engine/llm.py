from typing import TYPE_CHECKING

from google.genai import types

from rag_engine.config import PROJECT_ROOT

if TYPE_CHECKING:
    from google import genai


class LLM:
    def __init__(self, client: genai.Client, model: str = "gemma-4-31b-it") -> None:
        self.client = client
        self.model = model

    def load_prompt(self, name: str) -> str:
        path = PROJECT_ROOT / "src" / "rag_engine" / "query_processing" / "prompts" / f"{name}.md"

        return path.read_text().strip()

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        if not self.client:
            raise RuntimeError("No LLM configured")

        config = types.GenerateContentConfig(temperature=temperature)

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )

        if not response.text:
            raise ValueError("LLM returned no text")

        return response.text.strip()

    def prompt(self, template_name: str, **kwargs) -> str:  # noqa: ANN003
        template = self.load_prompt(template_name)
        return self.generate(template.format(**kwargs))
