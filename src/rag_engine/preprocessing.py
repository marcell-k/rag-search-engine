import string
from pathlib import Path

from nltk.stem import PorterStemmer

from rag_engine.data_loader import load_stop_words

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STOP_WORDS = PROJECT_ROOT / "data" / "stopwords.txt"

stemmer = PorterStemmer()
CACHED_STOP_WORDS = set(load_stop_words())


def clean_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tokenizer(text: str) -> list[str]:
    text = clean_text(text)
    tokens = [stemmer.stem(tok) for tok in text.split() if tok not in CACHED_STOP_WORDS]
    return tokens


def tokenize_first_term(term: str) -> str:
    token = tokenizer(term)
    if len(token) != 1:
        raise ValueError("only 1 token")
    return token[0]
