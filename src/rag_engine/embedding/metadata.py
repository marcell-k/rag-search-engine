import hashlib
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag_engine.embedding.chunker import RawChunk
    from rag_engine.models import ChunkMetadata

_NOTE_RE = re.compile(r"\bNote\s+(\d+)\b", re.IGNORECASE)
_DIGIT_RE = re.compile(r"\d")
_TOC_ROW_RE = re.compile(r"^\|.*\|\s*Page\b.*\d+\s*\|", re.IGNORECASE)
_BOILERPLATE_PHRASES: tuple[str, ...] = (
    "forward-looking statements",
    "actual results may differ materially",
    "undertakes no obligation to update",
    "undertake no obligation to update",
    "within the meaning of the private securities litigation reform act",
    "safe harbor",
    "except as required by law",
    "references to",
    "unless the context otherwise requires",
    "as used in this report",
    "incorporated herein by reference",
    "filed herewith",
    "furnished herewith",
    "pursuant to the requirements of section",
    "pursuant to the requirements of the securities exchange act",
    "the exhibits listed below",
    "index to exhibits",
    "signature",
    "signatures",
    "power of attorney",
    "report of independent registered public accounting firm",
)


def load_filing_header(path: Path) -> dict:
    with Path(path).open() as f:
        return json.load(f)


def compute_chunk_id(cik: str, accession_number: str, chunk_index: int) -> str:
    raw = f"{cik}:{accession_number}:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _is_boilerplate(chunk: RawChunk) -> bool:
    # TODO: The Token Jaccard Similarity Method ( compare current and previous year filing)
    # TODO: Semantic embedding
    text = chunk.text.lower()
    if any(phrase in text for phrase in _BOILERPLATE_PHRASES):
        return True
    if chunk.is_table:
        lines = chunk.text.splitlines()
        toc_rows = sum(1 for line in lines if _TOC_ROW_RE.match(line))
        if lines and toc_rows / len(lines) > 0.5:
            return True
    return False


def _contains_numbers(chunk: RawChunk) -> bool:
    return bool(_DIGIT_RE.search(chunk.text))


def _estimate_token_count(text: str) -> int:
    return len(text.split())


def attach_metadata(chunks: list[RawChunk], header: dict) -> list[ChunkMetadata]:
    res: list[ChunkMetadata] = []
    cik = header["cik"]
    accession_number = header["accession_number"]
    filing_date = header.get("filing_date") or header.get("period_end") or "UNKNOWN"

    for i, chunk in enumerate(chunks):
        note_match = _NOTE_RE.search(chunk.text[:60])
        note_number = note_match.group(1) if note_match else None
        is_footnote = bool(note_number) or bool(chunk.sec_title and "note" in chunk.sec_title.lower())

        metadata: ChunkMetadata = {
            "cik": cik,
            "accession_number": accession_number,
            "company_name": header["company_name"],
            "company_ticker": header["company_ticker"],
            "filing_type": header["filing_type"],
            "filing_date": filing_date,
            "period_of_report": header["period_end"],
            "period_type": header["period_type"],
            "is_amendment": header.get("is_amendment", False),
            "sec_item": chunk.sec_item or "",
            "sec_title": chunk.sec_title or "",
            "subsection_path": [],
            "note_number": note_number,
            "chunk_id": compute_chunk_id(cik, accession_number, i),
            "chunk_index": i,
            "token_count": _estimate_token_count(chunk.text),
            "topics": [],
            "is_table": chunk.is_table,
            "is_footnote": is_footnote,
            "is_boilerplate": _is_boilerplate(chunk),
            "contains_numbers": _contains_numbers(chunk),
            "start_char": getattr(chunk, "start_char", 0),
            "end_char": getattr(chunk, "end_char", len(chunk.text)),
        }
        res.append(metadata)
    return res
