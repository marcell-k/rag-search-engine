from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"

MOVIES_FILE = DATA_DIR / "movies.json"
STOPWORDS_FILE = DATA_DIR / "stopwords.txt"
GOLDEN_DATASET_FILE = DATA_DIR / "golden_dataset.json"

BM25_K1 = 1.5
BM25_B = 0.75

DEFAULT_SEARCH_LIMIT = 5
