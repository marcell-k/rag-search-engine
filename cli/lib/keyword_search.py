from cli.lib.utils import Movie, load_data, tokenizer


def search_command(query: str, n_results: int = 5) -> list[Movie]:
    data = load_data()
    query_tokens = tokenizer(query)
    res: list[Movie] = []
    for movie in data:
        movie_tokens = tokenizer(movie["title"])
        matching_token = movie_tokens & query_tokens
        if matching_token:
            res.append(movie)
        if len(res) == n_results:
            break
    return res
