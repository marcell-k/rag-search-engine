import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rag_engine.config import DATA_DIR, GOLDEN_DATASET_FILE, STOPWORDS_FILE


@dataclass
class FilingPaths:
    cik: str
    accession: str
    md_path: Path
    header_path: Path


def discover_filings(data_dir: Path = DATA_DIR) -> list[FilingPaths]:
    filings: list[FilingPaths] = []
    for cik_dir in sorted(data_dir.iterdir()):
        if not cik_dir.is_dir() or cik_dir.name.startswith("."):
            continue
        for accession_dir in sorted(cik_dir.iterdir()):
            if not accession_dir.is_dir():
                continue
            md_files = list(accession_dir.glob("*.md"))
            json_files = list(accession_dir.glob("*.json"))
            if not md_files or not json_files:
                continue
            filings.append(
                FilingPaths(
                    cik=cik_dir.name,
                    accession=accession_dir.name,
                    md_path=md_files[0],
                    header_path=json_files[0],
                )
            )
    return filings


@lru_cache(maxsize=1)
def load_golden() -> list:
    with Path(GOLDEN_DATASET_FILE).open() as f:
        data = json.load(f)
        return data["test_cases"]


@lru_cache(maxsize=1)
def load_stop_words() -> set[str]:
    with Path(STOPWORDS_FILE).open() as f:
        words = f.read().splitlines()
    return set(words)
