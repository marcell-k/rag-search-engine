import hashlib


def compute_chunk_id(cik: str, accession_number: str, chunk_index: int) -> str:
    raw = f"{cik}:{accession_number}{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()
