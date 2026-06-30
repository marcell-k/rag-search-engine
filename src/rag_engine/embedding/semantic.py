import re

import numpy as np
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


class SemanticChunk:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    def generate_embedding(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            raise ValueError("text is empty")
        return self.model.encode(text, convert_to_numpy=True)

    def chunk_text_semantically(self, text: str, percentile_threshold: int = 60, overlap: float = 0.2) -> list[str]:
        raw_splits = re.split(r"(?<=[.!?])\s+|\n+", text)
        sentences = [s.strip() for s in raw_splits if s.strip() and len(s.strip()) > 2]
        if len(sentences) <= 1:
            return sentences

        embeddings = self.model.encode(sentences, convert_to_numpy=True)

        distances = []
        for i in range(len(embeddings) - 1):
            distances.append(1 - cos_sim(embeddings[i], embeddings[i + 1]))

        threshold = np.percentile(distances, percentile_threshold)

        groups: list[list[str]] = []
        curr_chunk = [sentences[0]]

        for i, dist in enumerate(distances):
            if dist > threshold:
                groups.append(curr_chunk)
                curr_chunk = [sentences[i + 1]]
            else:
                curr_chunk.append(sentences[i + 1])
        if curr_chunk:
            groups.append(curr_chunk)
        chunks: list[str] = []
        for idx, group in enumerate(groups):
            if idx == 0:
                chunks.append(" ".join(group))
                continue
            prev_group = groups[idx - 1]
            n_overlap = max(1, round(len(prev_group) * overlap))
            overlap_sentences = prev_group[-n_overlap:]
            chunks.append(" ".join(overlap_sentences + group))
        return chunks
