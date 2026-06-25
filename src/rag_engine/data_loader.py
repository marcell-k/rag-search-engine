import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from rag_engine.config import GOLDEN_DATASET_FILE, MOVIES_FILE, STOPWORDS_FILE

if TYPE_CHECKING:
    from rag_engine.models import Movie


@lru_cache(maxsize=1)
def load_data() -> list[Movie]:
    with Path(MOVIES_FILE).open() as f:
        data = json.load(f)
        return data["movies"]


@lru_cache(maxsize=1)
def load_golden():
    with Path(GOLDEN_DATASET_FILE).open() as f:
        data = json.load(f)
        return data["test_cases"]


@lru_cache(maxsize=1)
def load_stop_words() -> set[str]:
    with Path(STOPWORDS_FILE).open() as f:
        words = f.read().splitlines()
    return set(words)
