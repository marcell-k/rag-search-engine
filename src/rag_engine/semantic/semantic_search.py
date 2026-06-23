import re

from rag_engine.data_loader import load_data
from rag_engine.semantic.embedder import ChunkedSemanticSearch, SemanticSearch


def embed_text(ss: SemanticSearch, text: str) -> None:
    token = ss.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {token[:3]}")
    print(f"dimensions: {token.shape[0]}")


def embed_query(query: str) -> None:
    ss = SemanticSearch()
    embedding = ss.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")


def fixed_size_chunking(text: str, chunk_size: int = 200, overlap: float = 0.2) -> list[str]:
    words = text.split()
    chunks = []
    overlap_words = int(chunk_size * overlap)
    step_size = chunk_size - overlap_words
    for i in range(0, len(words), step_size):
        chunk_words = words[i : i + chunk_size]
        if len(chunk_words) <= overlap_words:
            break
        chunks.append(" ".join(chunk_words))
    return chunks


def semantic_chunking(text: str, max_chunk_size: int = 200, overlap: float = 0.2) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks = []
    overlap_sentences = int(max_chunk_size * overlap)
    step_size = max_chunk_size - overlap_sentences
    for i in range(0, len(sentences), step_size):
        chunk_sentences = sentences[i : i + max_chunk_size]
        if i > 0 and len(chunk_sentences) <= overlap_sentences:
            break
        chunks.append(" ".join(chunk_sentences))
    return chunks


def search(ss: SemanticSearch, query: str, limit: int) -> None:
    movies = load_data()
    ss.load_or_create_embeddings(movies)
    results = ss.search(query, limit)
    for i, res in enumerate(results, start=1):
        print(f"{i}. {res['title']} (score: {res['score']:.4f}\n\t{res['description'][:80]}...")


def search_chunked(query: str, limit: int) -> None:
    css = ChunkedSemanticSearch()
    movies = load_data()
    css.load_or_create_chunk_embeddings(movies)
    results = css.search_chunks(query, limit)
    for i, res in enumerate(results, start=1):
        print(f"\n{i}. {res['title']} (score: {res['score']:.4f}")
        print(f"\t{res['description'][:40]}...")
