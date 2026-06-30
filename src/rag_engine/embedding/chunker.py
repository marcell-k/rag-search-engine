import hashlib

from rag_engine.config import DATA_DIR


def compute_chunk_id(cik: str, accession_number: str, chunk_index: int) -> str:
    raw = f"{cik}:{accession_number}:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()


DATA_PATH = DATA_DIR / "out.md"


def load_data() -> str:
    data = ""
    with DATA_PATH.open("r") as f:
        data = f.read()
    return data


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


def main() -> None:
    data = load_data()

    table_chunks, semantic_chunks = chunking(data)
    # print(table_chunks, len(table_chunks))
    print(len(semantic_chunks))
    print(table_chunks[0][:1])
    print(table_chunks[1][:3])


if __name__ == "__main__":
    main()
