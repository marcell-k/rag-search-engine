import json
from typing import TYPE_CHECKING

import numpy as np
from sentence_transformers import SentenceTransformer

from rag_engine.data_loader import load_data
from rag_engine.semantic.vector_db import VectorDB, cosine_similarity

if TYPE_CHECKING:
    from rag_engine.models import Movie, SearchResult


class SemanticSearch:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.vdb = VectorDB()
        self.document_map: dict[int, Movie] = {}

    def generate_embedding(self, text: str) -> np.ndarray:
        if not text or not text.strip():
            raise ValueError("text is empty")
        return self.model.encode(text, convert_to_numpy=True)

    def search(self, query: str, limit: int) -> list[dict]:
        query_embedding = self.generate_embedding(query)
        matches = self.vdb.search(query_embedding, limit)

        return [
            {
                "score": match["score"],
                "title": self.document_map[match["id"]]["title"],
                "description": self.document_map[match["id"]]["description"],
            }
            for match in matches
        ]

    def build_embeddings(self, documents: list[Movie]) -> np.ndarray:
        movie_strings = []
        doc_ids = []

        for doc in documents:
            self.document_map[doc["id"]] = doc
            doc_ids.append(doc["id"])
            movie_strings.append(f"{doc['title']}: {doc['description']}")

        embeddings = self.model.encode(movie_strings, show_progress_bar=True, convert_to_numpy=True)

        self.vdb.build(doc_ids, embeddings)
        self.vdb.save()

        return embeddings

    def load_or_create_embeddings(self, documents: list[Movie]) -> np.ndarray | None:
        for doc in documents:
            self.document_map[doc["id"]] = doc

        try:
            self.vdb.load()
            return self.vdb.embeddings
        except FileNotFoundError:
            return self.build_embeddings(documents)


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings: np.ndarray | None = None
        self.chunk_metadata: list[dict] | None = None
        self.documents: list[Movie] = []

    def build_chunk_embeddings(self, documents: list[Movie]) -> np.ndarray:
        from rag_engine.config import CACHE_DIR
        from rag_engine.semantic.search import semantic_chunking

        self.documents = documents
        self.document_map = {}
        all_chunks = []
        chunk_metadata = []

        for movie_idx, doc in enumerate(documents):
            self.document_map[doc["id"]] = doc

            if not doc.get("description"):
                continue

            doc_chunks = semantic_chunking(doc["description"], max_chunk_size=4, overlap=0.25)

            for chunk_idx, chunk in enumerate(doc_chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({"movie_idx": movie_idx, "chunk_idx": chunk_idx, "total_chunks": len(doc_chunks)})

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)
        self.chunk_metadata = chunk_metadata

        CACHE_DIR.mkdir(parents=True, exist_ok=True)

        np.save(CACHE_DIR / "chunk_embeddings.npy", self.chunk_embeddings)
        with (CACHE_DIR / "chunk_metadata.json").open("w") as f:
            json.dump({"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[Movie]) -> np.ndarray | None:
        from rag_engine.config import CACHE_DIR

        self.documents = documents
        self.document_map = {doc["id"]: doc for doc in documents}

        embeddings_file = CACHE_DIR / "chunk_embeddings.npy"
        metadata_file = CACHE_DIR / "chunk_metadata.json"

        if embeddings_file.exists() and metadata_file.exists():
            self.chunk_embeddings = np.load(embeddings_file)
            with metadata_file.open("r") as f:
                data = json.load(f)
                self.chunk_metadata = data.get("chunks", [])
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 5) -> list[SearchResult]:
        if self.chunk_embeddings is None or not self.chunk_metadata:
            return []

        # 1. Generate the 1D embedding for the query
        query_embedding = self.generate_embedding(query)

        # 2. Pass the entire 2D matrix of chunk embeddings at once
        similarities = cosine_similarity(self.chunk_embeddings, query_embedding)

        # 3. Sort all similarities at once and grab the top indices
        sorted_indices = np.argsort(similarities)[::-1]
        results = []
        seen_docs: set[int] = set()

        for idx in sorted_indices:
            metadata = self.chunk_metadata[idx]
            movie_idx = metadata["movie_idx"]
            doc = self.documents[movie_idx]
            doc_id = doc["id"]
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            results.append(
                {
                    "doc_id": doc_id,
                    "chunk_idx": metadata["chunk_idx"],
                    "score": float(similarities[idx]),
                    "title": doc["title"],
                    "description": doc["description"],
                }
            )

            if len(results) == limit:
                break

        return results


def verify_model(ss: SemanticSearch) -> tuple[SentenceTransformer, int | None]:
    return ss.model, ss.model.max_seq_length


def verify_embeddings(ss: SemanticSearch) -> None:
    movies = load_data()
    embeddings = ss.load_or_create_embeddings(movies)
    if ss.document_map and embeddings is not None:
        print(f"Number of docs:   {len(ss.document_map)}")
        print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
