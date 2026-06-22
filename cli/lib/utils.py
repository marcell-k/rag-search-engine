import json
import math
import pickle
import string
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

from nltk.stem import PorterStemmer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "movies.json"
STOP_WORDS = PROJECT_ROOT / "data" / "stopwords.txt"
stemmer = PorterStemmer()


class Movie(TypedDict):
    id: int
    title: str
    description: str


class InvertedIndex:
    def __init__(self) -> None:
        self.inverted_index: dict[str, set[int]] = defaultdict(set)
        self.docmap: dict[int, Movie] = {}
        self.term_frequencies = defaultdict(Counter)

        self._cache_dir = PROJECT_ROOT / "cache"
        self._index_file = self._cache_dir / "inverted_index.pkl"
        self._docmap_file = self._cache_dir / "docmap.pkl"
        self._term_frequencies_file = self._cache_dir / "term_frequencies.pkl"

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenizer(text)
        for tok in tokens:
            self.inverted_index[tok].add(doc_id)
        self.term_frequencies[doc_id].update(tokens)

    def get_tf(self, doc_id: int, term: str) -> int:
        token = tokenize_first_term(term)
        return self.term_frequencies[doc_id][token]

    def get_documents(self, token: str) -> list[int]:
        return sorted(self.inverted_index[token])

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

    def load(self) -> None:
        if not self._index_file.exists() or not self._docmap_file.exists() or not self._term_frequencies_file.exists():
            raise FileNotFoundError("inverted_index or docmap not found")
        with self._index_file.open("rb") as f:
            self.inverted_index = pickle.load(f)
        with self._docmap_file.open("rb") as f:
            self.docmap = pickle.load(f)
        with self._term_frequencies_file.open("rb") as f:
            self.term_frequencies = pickle.load(f)


def build_command() -> None:
    ii = InvertedIndex()
    ii.build()
    ii.save()


def tf_command(ii: InvertedIndex, doc_id: int, term: str) -> int:
    return ii.get_tf(doc_id, term)


def idf_command(ii: InvertedIndex, term: str) -> float:
    token = tokenize_first_term(term)
    term_match_doc_count = len(ii.get_documents(token))
    total_doc_count = len(ii.docmap)
    idf_value = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
    return idf_value


def tfidf_command(ii: InvertedIndex, doc_id: int, term: str) -> float:
    idf_value = idf_command(ii, term)
    tf_value = tf_command(ii, doc_id, term)
    return idf_value * tf_value


def tokenize_first_term(term: str) -> str:
    token = tokenizer(term)
    if len(token) != 1:
        raise ValueError("only 1 token")
    return token[0]


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


def clean_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenizer(text: str) -> list[str]:
    text = clean_text(text)
    stop_words = load_stop_words()
    tokens = [stemmer.stem(tok) for tok in text.split() if tok not in stop_words]
    return tokens
