import json
import string
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


def tokenizer(text: str) -> set[str]:
    text = clean_text(text)
    stop_words = load_stop_words()
    tokens = {stemmer.stem(tok) for tok in text.split() if tok not in stop_words}
    return tokens
