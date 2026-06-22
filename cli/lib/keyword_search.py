from cli.lib.utils import InvertedIndex, Movie, tokenizer


def search_command(query: str, n_results: int = 5) -> list[Movie]:
    res: list[Movie] = []
    seen: set[int] = set()
    inverted_index = InvertedIndex()
    inverted_index.load()

    query_tokens = tokenizer(query)
    for token in query_tokens:
        matching_doc_ids = inverted_index.get_documents(token)
        for doc_id in matching_doc_ids:
            if doc_id not in seen:
                res.append(inverted_index.docmap[doc_id])
                seen.add(doc_id)
            if len(res) == n_results:
                break
    return res
