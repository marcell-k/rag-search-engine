import string
from functools import lru_cache
from pathlib import Path

from nltk.stem import PorterStemmer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STOP_WORDS = PROJECT_ROOT / "data" / "stopwords.txt"

stemmer = PorterStemmer()


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


def tokenize_first_term(term: str) -> str:
    token = tokenizer(term)
    if len(token) != 1:
        raise ValueError("only 1 token")
    return token[0]
