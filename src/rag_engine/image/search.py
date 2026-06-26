from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from rag_engine.semantic.vector_db import cosine_similarity

if TYPE_CHECKING:
    from rag_engine.models import Movie


class MultimodalSearch:
    def __init__(self, documents: list[Movie], model_name: str = "clip-ViT-B-32") -> None:
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = []
        for doc in documents:
            self.texts.append(f"{doc['title']}: {doc['description']}")

        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True, convert_to_numpy=True)

    def search(self, image_path: str, limit: int = 5) -> list[dict[str, str | int | float]]:
        image_emb = self.embed_image(image_path)
        similarities = cosine_similarity(self.text_embeddings, image_emb)

        top_indices = np.argsort(similarities)[::-1][:limit]
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            results.append(
                {
                    "title": doc["title"],
                    "description": doc["description"],
                    "doc_id": idx,
                    "score": float(similarities[idx]),
                }
            )

        return results

    def embed_image(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path)
        return self.model.encode([img], convert_to_numpy=True)[0]

    def verify_image_embedding(self, image_path: str) -> None:
        embedding = self.embed_image(image_path)
        print(f"Embedding shape: {embedding.shape[0]} dimensions")
