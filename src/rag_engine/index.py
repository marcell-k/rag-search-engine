import json
import math
import pickle
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

from rag_engine.preprocessing import tokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "movies.json"
STOP_WORDS = PROJECT_ROOT / "data" / "stopwords.txt"

BM25_K1 = 1.5
BM25_B = 0.75


class Movie(TypedDict):
    id: int
    title: str
    description: str


class InvertedIndex:
    def __init__(self) -> None:
        self.inverted_index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, Movie] = {}
        self.term_frequencies = defaultdict(Counter)
        self.doc_lengths: dict[int, int] = {}

        self.avg_doc_length = 0.0

        self._cache_dir = PROJECT_ROOT / "cache"
        self._index_file = self._cache_dir / "inverted_index.pkl"
        self._docmap_file = self._cache_dir / "docmap.pkl"
        self._term_frequencies_file = self._cache_dir / "term_frequencies.pkl"
        self._doc_lengths = self._cache_dir / "doc_lengths.pkl"

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenizer(text)
        for tok in tokens:
            self.inverted_index[tok].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)
        self.doc_lengths[doc_id] = len(tokens)

    def get_tf(self, doc_id: int, token: str) -> int:
        return self.term_frequencies[doc_id][token]

    def get_documents(self, token: str) -> list[int]:
        return sorted(self.inverted_index[token])

    def get_bm25_idf(self, token: str) -> float:
        N = len(self.docmap)
        df = len(self.get_documents(token))
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id: int, token: str) -> float:
        tf = self.get_tf(doc_id, token)
        length_norm = 1 - BM25_B + BM25_B * (self.doc_lengths[doc_id] / self.avg_doc_length)
        return tf * (BM25_K1 + 1) / (tf + BM25_K1 * length_norm)

    def bm25(self, doc_id: int, token: str) -> float:
        return self.get_bm25_tf(doc_id, token) * self.get_bm25_idf(token)

    def build(self) -> None:
        movies = load_data()
        for movie in movies:
            self.__add_document(movie["id"], f"{movie['title']} {movie['description']}")
            self.docmap[movie["id"]] = movie

    def save(self) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        with self._index_file.open("wb") as f:
            pickle.dump(self.inverted_index, f)
        with self._docmap_file.open("wb") as f:
            pickle.dump(self.docmap, f)
        with self._term_frequencies_file.open("wb") as f:
            pickle.dump(self.term_frequencies, f)
        with self._doc_lengths.open("wb") as f:
            pickle.dump(self.doc_lengths, f)
        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def load(self) -> None:
        if (
            not self._index_file.exists()
            or not self._docmap_file.exists()
            or not self._term_frequencies_file.exists()
            or not self._doc_lengths.exists()
        ):
            raise FileNotFoundError("inverted_index or docmap not found")
        with self._index_file.open("rb") as f:
            self.inverted_index = pickle.load(f)  # noqa: S301
        with self._docmap_file.open("rb") as f:
            self.docmap = pickle.load(f)  # noqa: S301
        with self._term_frequencies_file.open("rb") as f:
            self.term_frequencies = pickle.load(f)  # noqa: S301
        with self._doc_lengths.open("rb") as f:
            self.doc_lengths = pickle.load(f)  # noqa: S301
        if self.doc_lengths:
            self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)


@lru_cache(maxsize=1)
def load_data() -> list[Movie]:
    with Path(DATA_PATH).open() as f:
        data = json.load(f)
        return data["movies"]


@lru_cache(maxsize=1)
def load_stop_words() -> set[str]:
    with Path(STOP_WORDS).open() as f:
        words = f.read().splitlines()
    return set(words)
