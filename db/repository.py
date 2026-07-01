from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np

    from db.client import DatabaseClient
    from rag_engine.models import ChunkMetadata

BATCH_SIZE = 500

_UPSERT_SQL = """
INSERT INTO chunks (
    chunk_id, chunk_index, cik, company_name, company_ticker,
    filing_type, filing_date, period_of_report, period_type,
    accession_number, is_amendment, sec_item, sec_title,
    subsection_path, note_number, content, token_count,
    is_table, is_footnote, is_boilerplate, contains_numbers,
    topics, embedding
) VALUES (
    %(chunk_id)s, %(chunk_index)s, %(cik)s, %(company_name)s, %(company_ticker)s,
    %(filing_type)s, %(filing_date)s, %(period_of_report)s, %(period_type)s,
    %(accession_number)s, %(is_amendment)s, %(sec_item)s, %(sec_title)s,
    %(subsection_path)s, %(note_number)s, %(content)s, %(token_count)s,
    %(is_table)s, %(is_footnote)s, %(is_boilerplate)s, %(contains_numbers)s,
    %(topics)s, %(embedding)s
)
ON CONFLICT (chunk_id) DO UPDATE SET
    chunk_index       = EXCLUDED.chunk_index,
    company_name      = EXCLUDED.company_name,
    company_ticker    = EXCLUDED.company_ticker,
    filing_type       = EXCLUDED.filing_type,
    filing_date       = EXCLUDED.filing_date,
    period_of_report  = EXCLUDED.period_of_report,
    period_type       = EXCLUDED.period_type,
    is_amendment      = EXCLUDED.is_amendment,
    sec_item          = EXCLUDED.sec_item,
    sec_title         = EXCLUDED.sec_title,
    subsection_path   = EXCLUDED.subsection_path,
    note_number       = EXCLUDED.note_number,
    content           = EXCLUDED.content,
    token_count       = EXCLUDED.token_count,
    is_table          = EXCLUDED.is_table,
    is_footnote       = EXCLUDED.is_footnote,
    is_boilerplate    = EXCLUDED.is_boilerplate,
    contains_numbers  = EXCLUDED.contains_numbers,
    topics            = EXCLUDED.topics,
    embedding         = EXCLUDED.embedding,
    updated_at        = NOW()
"""


def _to_row(meta: ChunkMetadata, content: str, embedding: np.ndarray) -> dict:
    """Flatten metadata + content + embedding into the param dict the upsert SQL expects."""
    return {
        "chunk_id": meta["chunk_id"],
        "chunk_index": meta["chunk_index"],
        "cik": meta["cik"],
        "company_name": meta["company_name"],
        "company_ticker": meta["company_ticker"],
        "filing_type": meta["filing_type"],
        "filing_date": meta["filing_date"],
        "period_of_report": meta["period_of_report"],
        "period_type": meta["period_type"],
        "accession_number": meta["accession_number"],
        "is_amendment": meta["is_amendment"],
        "sec_item": meta["sec_item"],
        "sec_title": meta["sec_title"],
        "subsection_path": meta["subsection_path"],
        "note_number": meta["note_number"],
        "content": content,
        "token_count": meta["token_count"],
        "is_table": meta["is_table"],
        "is_footnote": meta["is_footnote"],
        "is_boilerplate": meta["is_boilerplate"],
        "contains_numbers": meta["contains_numbers"],
        "topics": meta["topics"],
        "embedding": embedding.tolist(),
    }


def _batched(rows: Sequence[dict], size: int) -> list[list[dict]]:
    return [list(rows[i : i + size]) for i in range(0, len(rows), size)]


class ChunkRepository:
    def __init__(self, client: DatabaseClient) -> None:
        self.client = client

    @staticmethod
    def build_rows(
        metadatas: Sequence[ChunkMetadata], contents: Sequence[str], embeddings: Sequence[np.ndarray]
    ) -> list[dict]:
        return [
            _to_row(meta, content, embedding)
            for meta, content, embedding in zip(metadatas, contents, embeddings, strict=True)
        ]

    async def upsert_chunks(self, rows: Sequence[dict], batch_size: int = BATCH_SIZE) -> int:
        """Batch upsert chunk rows."""
        if not rows:
            return 0

        async with self.client.connection() as conn:
            async with conn.cursor() as cur:
                for batch in _batched(rows, batch_size):
                    await cur.executemany(_UPSERT_SQL, batch)
            await conn.commit()

        return len(rows)
