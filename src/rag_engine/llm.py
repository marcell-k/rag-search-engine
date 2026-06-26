from typing import TYPE_CHECKING

from google.genai import types

from rag_engine.config import PROJECT_ROOT

if TYPE_CHECKING:
    from google import genai


class LLM:
    def __init__(self, client: genai.Client, model: str = "gemma-4-31b-it", system_instruction: str = "") -> None:
        self.client = client
        self.model = model
        self.system_instruction = system_instruction

    def load_prompt(self, name: str) -> str:
        path = PROJECT_ROOT / "src" / "rag_engine" / "query_processing" / "prompts" / f"{name}.md"

        return path.read_text().strip()

    def generate(self, prompt: str, temperature: float = 0.0) -> str:
        if not self.client:
            raise RuntimeError("No LLM configured")

        config = types.GenerateContentConfig(temperature=temperature, system_instruction=self.system_instruction)

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )

        text = response.text
        if not text:
            candidate = response.candidates[0] if response.candidates else None
            reason = getattr(getattr(candidate, "finish_reason", None), "name", "unknown")
            raise ValueError(f"LLM returned no text (finish_reason={reason})")
        return text.strip()

    def prompt(self, template_name: str, **kwargs) -> str:  # noqa: ANN003
        template = self.load_prompt(template_name)
        return self.generate(template.format(**kwargs))
