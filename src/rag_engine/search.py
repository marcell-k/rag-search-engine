from typing import TYPE_CHECKING

from rag_engine.preprocessing import tokenizer

if TYPE_CHECKING:
    from rag_engine.index import InvertedIndex, Movie


def search_command(ii: InvertedIndex, query: str, n_results: int = 5) -> list[Movie]:
    res: list[Movie] = []
    seen: set[int] = set()

    query_tokens = tokenizer(query)
    for token in query_tokens:
        matching_doc_ids = ii.get_documents(token)
        for doc_id in matching_doc_ids:
            if doc_id not in seen:
                res.append(ii.docmap[doc_id])
                seen.add(doc_id)
            if len(res) == n_results:
                break
    return res
