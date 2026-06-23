import pickle
from typing import TypedDict

import numpy as np

from rag_engine.config import CACHE_DIR


class VectorMatch(TypedDict):
    id: int
    score: float


class VectorDB:
    def __init__(self) -> None:
        self.embeddings: np.ndarray | None = None
        self.doc_ids: list[int] = []

        self._cache_dir = CACHE_DIR
        self._embeddings_file = self._cache_dir / "vector_db_embeddings.npy"
        self._doc_ids_file = self._cache_dir / "vector_db_ids.pkl"

    def build(self, doc_ids: list[int], embeddings: np.ndarray) -> None:
        if len(doc_ids) != embeddings.shape[0]:
            raise ValueError("Number of doc_ids must match the number of embeddings.")
        self.doc_ids = doc_ids
        self.embeddings = embeddings

    def search(self, query_vector: np.ndarray, limit: int = 5) -> list[VectorMatch]:
        if self.embeddings is None or not self.doc_ids:
            return []

        similarities = cosine_similarity(query_vector, self.embeddings)

        top_indices = np.argsort(similarities)[::-1][:limit]

        return [{"id": self.doc_ids[idx], "score": float(similarities[idx])} for idx in top_indices]

    def save(self) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        if self.embeddings is not None:
            np.save(self._embeddings_file, self.embeddings)

        with self._doc_ids_file.open("wb") as f:
            pickle.dump(self.doc_ids, f)

    def load(self) -> None:
        if not self._embeddings_file.exists() or not self._doc_ids_file.exists():
            raise FileNotFoundError("Vector DB files not found. Build and save the DB first.")

        self.embeddings = np.load(self._embeddings_file)
        with self._doc_ids_file.open("rb") as f:
            self.doc_ids = pickle.load(f)  # noqa: S301


def cosine_similarity(embeddings: np.ndarray, query_vector: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity between a matrix of embeddings and a single query vector."""
    dot_product = np.dot(embeddings, query_vector)
    norm_matrix = np.linalg.norm(embeddings, axis=1)
    norm_query = np.linalg.norm(query_vector)

    norms = norm_matrix * norm_query
    norms[norms == 0] = 1.0  # Avoid division by zero

    return dot_product / norms
