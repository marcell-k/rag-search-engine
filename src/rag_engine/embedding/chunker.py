import asyncio
import re
from dataclasses import dataclass

import numpy as np

from db.client import DatabaseClient
from db.repository import ChunkRepository
from rag_engine.data_loader import discover_filings
from rag_engine.embedding.metadata import attach_metadata, load_filing_header
from rag_engine.embedding.semantic import SemanticChunk
from rag_engine.embedding.topics import assign_topics
from rag_engine.filing_types import build_item_title_lookup

_ITEM_RE = re.compile(r"^#{1,3}\s*ITEM\s+(\d+[A-Z]?(?:\.\d+)?)\.?\s*(.*)$", re.IGNORECASE)
_ITEM_TITLES = build_item_title_lookup()
_HEADING_RE = re.compile(r"^(#{1,3})\s+\S")


MIN_CHUNK_CHARS = 30
TEXT_CHUNK_OVERLAP = 0.2
PERCENTILE_THRESHOLD = 60


@dataclass
class RawChunk:
    text: str
    is_table: bool
    sec_item: str | None
    sec_title: str | None


def _match_item_header(line: str) -> tuple[str, str | None] | None:
    match = _ITEM_RE.match(line)
    if not match:
        return None
    item_key = f"ITEM {match.group(1).upper()}"
    heading_text = match.group(2).strip()
    sec_title = _ITEM_TITLES.get(item_key, heading_text or None)
    return item_key, sec_title


def _is_unmatched_heading(line: str) -> bool:
    match = _HEADING_RE.match(line)
    if match is None or _match_item_header(line) is not None:
        return False
    return len(match.group(1)) <= 2


def _merge_short_pieces(pieces: list[str], min_chars: int) -> list[str]:
    """Merge any piece under `min_chars` into its predecessor."""
    if not pieces:
        return pieces
    merged: list[str] = [pieces[0]]
    for piece in pieces[1:]:
        if len(piece) < min_chars or len(merged[-1]) < min_chars:
            merged[-1] = f"{merged[-1]} {piece}".strip()
        else:
            merged.append(piece)
    return merged


def _flush_text_buffer(
    buffer: list[str],
    sec_item: str | None,
    sec_title: str | None,
    semantic_chunker: SemanticChunk,
    chunks: list[RawChunk],
) -> None:
    """Run accumulated prose lines through semantic chunking and append the results."""
    text = "\n".join(buffer).strip()
    buffer.clear()
    if not text:
        return
    pieces = semantic_chunker.chunk_text_semantically(
        text, percentile_threshold=PERCENTILE_THRESHOLD, overlap=TEXT_CHUNK_OVERLAP
    )
    for piece in _merge_short_pieces(pieces, MIN_CHUNK_CHARS):
        chunk = RawChunk(text=piece, is_table=False, sec_item=sec_item, sec_title=sec_title)
        chunks.append(chunk)


def build_filing_chunks(text: str, semantic_chunker: SemanticChunk) -> list[RawChunk]:
    chunks: list[RawChunk] = []
    text_buffer: list[str] = []
    table_buffer: list[str] = []
    in_table = False

    current_item: str | None = None
    current_title: str | None = None

    def flush_text() -> None:
        _flush_text_buffer(text_buffer, current_item, current_title, semantic_chunker, chunks)

    def flush_table() -> None:
        if table_buffer:
            chunks.append(
                RawChunk(text="\n".join(table_buffer), is_table=True, sec_item=current_item, sec_title=current_title)
            )
            table_buffer.clear()

    for line in text.split("\n"):
        is_table_line = line.startswith("|")

        if is_table_line:
            if not in_table:
                flush_text()
                in_table = True
            table_buffer.append(line)
            continue

        if in_table:
            flush_table()
            in_table = False

        header = _match_item_header(line)
        if header is not None:
            flush_text()
            current_item, current_title = header
            continue

        if _is_unmatched_heading(line):
            flush_text()
            current_item, current_title = None, None
            continue

        text_buffer.append(line)

    if in_table:
        flush_table()
    else:
        flush_text()

    return chunks


def chunking(text: str) -> tuple[list[list[str]], list[str]]:
    semantic_chunks: list[str] = []
    table_chunks: list[list[str]] = []
    lines = text.split("\n")
    start_table = True
    for line in lines:
        if line.startswith("|") and start_table:
            table_chunks.append([])
            table_chunks[-1].append(line)
            start_table = False

        elif line.startswith("|") and not start_table:
            table_chunks[-1].append(line)
            start_table = False
        else:
            start_table = True
            semantic_chunks.append(line)
    return table_chunks, semantic_chunks


def embed_chunks(chunks: list[str], sc: SemanticChunk) -> np.ndarray:
    if not chunks:
        return np.empty((0, 384))
    return sc.model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)


async def run() -> None:
    semantic_chunker = SemanticChunk()
    client = DatabaseClient()
    await client.connect()
    try:
        repo = ChunkRepository(client)
        for filing in discover_filings():
            data = filing.md_path.read_text()
            header = load_filing_header(filing.header_path)

            chunks = build_filing_chunks(data, semantic_chunker)

            # n_tables = sum(c.is_table for c in chunks)
            # print(f"{len(chunks)} chunks ({n_tables} tables, {len(chunks) - n_tables} text)")
            # for c in chunks[:5]:
            #     print(f"[{c.sec_item}] is_table={c.is_table} len={len(c.text)}")
            #
            # r = []
            # rtable = []
            # rtext = []
            # for c in chunks:
            #     r.append(len(c.text))
            #     if c.is_table:
            #         rtable.append(len(c.text))
            #     else:
            #         rtext.append(len(c.text))
            #
            # def print_stat(data: list[int]) -> None:
            #     print(f"Mean:   {np.mean(data):.2f}")
            #     print(f"Median: {np.median(data):.2f}")
            #     print(f"Std:    {np.std(data):.2f}")
            #     quantiles = np.quantile(data, [i / 10 for i in range(1, 10)])
            #     print("Deciles:", " ".join(f"{q:.2f}" for q in quantiles))
            #     print("\nDistribution:")
            #     counts, bin_edges = np.histogram(data, bins=10)
            #     max_count = max(counts) if max(counts) > 0 else 1
            #     max_bar_width = 30
            #     for i in range(len(counts)):
            #         bar_length = int((counts[i] / max_count) * max_bar_width)
            #         bar = "█" * bar_length
            #         bin_label = f"[{bin_edges[i]:6.2f} : {bin_edges[i + 1]:6.2f}]"
            #         print(f"  {bin_label} | {bar:<{max_bar_width}} ({counts[i]})")
            #
            # print("Total")
            # print_stat(r)
            # print("---\n")
            # print("Table")
            # print_stat(rtable)
            # print("---\n")
            # print("Text")
            # print_stat(rtext)
            # print("---\n")

            meta = attach_metadata(chunks, header)
            topics = []
            for chunk in chunks:
                chunk_embedding = semantic_chunker.generate_embedding(chunk.text)
                topics.append(assign_topics(chunk.sec_item, chunk_embedding, semantic_chunker))
            for m, t in zip(meta, topics, strict=True):
                m["topics"] = t

            # print("\nTags per Chunk Distribution:")
            # tag_counts = [len(t) for t in topics]
            # tag_distribution = Counter(tag_counts)
            # if tag_distribution:
            #     max_tags = max(tag_distribution.keys())
            #     max_freq = max(tag_distribution.values())
            #     max_bar = 30
            #     for num_tags in range(max_tags + 1):
            #         count = tag_distribution.get(num_tags, 0)
            #         bar_length = int((count / max_freq) * max_bar) if max_freq > 0 else 0
            #         bar = "█" * bar_length
            #         label = f"{num_tags} tag{'s' if num_tags != 1 else ' '}"
            #         print(f"  {label:7} | {bar:<{max_bar}} ({count})")
            # print("---\n")
            #
            # print("Total tags assigned:", Counter(tag for chunk_topics in topics for tag in chunk_topics))
            # print("Total chunks:", len(meta))
            # print("Boilerplate chunks:", sum(1 for x in meta if x["is_boilerplate"]))
            # print("Table chunks:", sum(1 for x in meta if x["is_table"]))
            # print("Footnote chunks:", sum(1 for x in meta if x["is_footnote"]))

            chunk_texts = [c.text for c in chunks]
            embeddings = embed_chunks(chunk_texts, semantic_chunker)
            # print(f"Embeddings shape: {embeddings.shape}")

            if len(meta) != len(chunk_texts) or len(meta) != embeddings.shape[0]:
                raise ValueError(f"count mismatch: {len(meta)} vs {len(chunk_texts)} vs {embeddings.shape[0]}")

            rows = repo.build_rows(meta, chunk_texts, list(embeddings))
            n_written = await repo.upsert_chunks(rows)
            print(f"[{filing.cik}/{filing.accession}] upserted {n_written} chunks")
    finally:
        await client.disconnect()


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
