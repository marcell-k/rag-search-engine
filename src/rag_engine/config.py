from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

GOLDEN_DATASET_FILE = DATA_DIR / "golden_dataset.json"
