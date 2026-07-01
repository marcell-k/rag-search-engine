import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return dot / norm


def cosine_similarity_batch(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    dot = matrix @ query
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    norms[norms == 0] = 1.0
    return dot / norms
