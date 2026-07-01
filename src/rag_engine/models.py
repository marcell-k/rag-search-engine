from typing import TYPE_CHECKING, Literal, TypedDict

if TYPE_CHECKING:
    from rag_engine.filing_types import FilingType


class SearchResult(TypedDict):
    chunk_id: str
    score: float
    content: str
    cik: str
    company_name: str
    filing_type: FilingType
    period_of_report: str | None
    sec_item: str
    sec_title: str


class HybridSearchResult(SearchResult, total=True):
    fts_score: float
    vec_score: float
    hybrid_score: float

    fts_rank: int | None
    vec_rank: int | None
    hybrid_rank: int | None


type TopicTag = Literal[
    "revenue_recognition",
    "leases",
    "debt",
    "income_taxes",
    "stock_based_compensation",
    "business_combinations",
    "goodwill_and_intangibles",
    "commitments_and_contingencies",
    "segment_reporting",
    "subsequent_events",
    "fair_value_measurements",
    "derivatives_and_hedging",
    "pension_and_benefits",
    "related_party_transactions",
    "risk_factors",
    "legal_proceedings",
    "internal_controls",
    "executive_compensation",
    "cybersecurity",
    "liquidity_and_capital_resources",
    "other",
]

type PeriodType = Literal["FY", "Q1", "Q2", "Q3", "Q4"]


class ChunkMetadata(TypedDict):
    # Entity Identifiers
    cik: str
    company_ticker: list[str] | None
    company_name: str

    # Filing Identifiers
    filing_type: FilingType
    period_of_report: str | None  # Format: YYYY-MM-DD (End of Q1, Q2, Q3, or FY)
    period_type: PeriodType | None
    accession_number: str
    is_amendment: bool  # True if filing_type ends in '/A'

    # Document Hierarchy
    sec_item: str
    sec_title: str
    subsection_path: list[str]
    note_number: str | None

    # Chunk Specifics
    chunk_id: str  # hash (cik + chunk_index)
    chunk_index: int
    token_count: int

    # Enrichment
    topics: list[TopicTag]

    # Flattened Vector DB Flags
    is_table: bool
    is_footnote: bool
    is_boilerplate: bool
    contains_numbers: bool

    # Citations
    start_char: int
    end_char: int
