from typing import TypedDict


class Movie(TypedDict):
    id: int
    title: str
    description: str


class SearchResult(TypedDict):
    doc_id: int
    score: float
    title: str
    description: str


class HybridSearchResult(SearchResult, total=True):
    bm_score: float
    sem_score: float
    hybrid_score: float

    bm_rank: int | None
    sem_rank: int | None
    hybrid_rank: int | None
