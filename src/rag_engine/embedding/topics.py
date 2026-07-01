from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np

from rag_engine.embedding.utils import cosine_similarity_batch

if TYPE_CHECKING:
    from rag_engine.embedding.semantic import SemanticChunk
    from rag_engine.models import TopicTag

SIMILARITY_THRESHOLD = 0.35
_ITEM_TOPIC_MAP: dict[str, TopicTag] = {
    "ITEM 1A": "risk_factors",
    "ITEM 1C": "cybersecurity",
    "ITEM 3": "legal_proceedings",
    "ITEM 9A": "internal_controls",
    "ITEM 11": "executive_compensation",
}

_AMBIGUOUS_ITEMS = {"ITEM 7", "ITEM 7A", "ITEM 8"}
_TOPIC_DESCRIPTIONS: dict[TopicTag, str] = {
    "revenue_recognition": "disclosures about how and when the company recognizes revenue from contracts with customers",  # noqa: E501
    "leases": "disclosures about operating and finance leases, right-of-use assets, and lease liabilities",
    "debt": "disclosures about long-term debt, credit facilities, notes payable, and debt covenants",
    "income_taxes": "disclosures about income tax expense, deferred taxes, and uncertain tax positions",
    "stock_based_compensation": "disclosures about stock options, restricted stock units, and equity compensation plans",  # noqa: E501
    "business_combinations": "disclosures about acquisitions, mergers, and business combination accounting",
    "goodwill_and_intangibles": "disclosures about goodwill, trademarks, and intangible asset impairment testing",
    "commitments_and_contingencies": "disclosures about contractual commitments, guarantees, and contingent liabilities",  # noqa: E501
    "segment_reporting": "disclosures about operating segments, geographic regions, and segment profitability",
    "subsequent_events": "disclosures about events occurring after the balance sheet date but before filing",
    "fair_value_measurements": "disclosures about fair value hierarchy levels and valuation techniques for assets and liabilities",  # noqa: E501
    "derivatives_and_hedging": "disclosures about derivative instruments, hedging activities, and notional amounts",
    "pension_and_benefits": "disclosures about pension plans, postretirement benefits, and benefit obligations",
    "related_party_transactions": "disclosures about transactions with related parties, affiliates, or officers",
    "risk_factors": "discussion of risks and uncertainties that could affect the company's business",
    "legal_proceedings": "discussion of litigation, lawsuits, and legal proceedings involving the company",
    "internal_controls": "discussion of internal controls over financial reporting and disclosure controls",
    "executive_compensation": "disclosures about executive officer and director compensation",
    "cybersecurity": "discussion of cybersecurity risk management and material cybersecurity incidents",
    "liquidity_and_capital_resources": "discussion of cash flows, liquidity, and capital resource availability",
}


@lru_cache(maxsize=1)
def _topic_embeddings(semantic_chunker: SemanticChunk) -> tuple[list[TopicTag], np.ndarray]:
    labels = list(_TOPIC_DESCRIPTIONS.keys())
    vectors = np.stack([semantic_chunker.generate_embedding(_TOPIC_DESCRIPTIONS[label]) for label in labels])
    return labels, vectors


def classify_by_item(sec_item: str | None) -> list[TopicTag] | None:
    if sec_item is None:
        return None
    topic = _ITEM_TOPIC_MAP.get(sec_item)
    return [topic] if topic else None


def classify_by_embedding(
    chunk_embedding: np.ndarray, semantic_chunker: SemanticChunk, threshold: float = SIMILARITY_THRESHOLD
) -> list[TopicTag]:
    labels, topic_vectors = _topic_embeddings(semantic_chunker)
    sim = cosine_similarity_batch(topic_vectors, chunk_embedding)
    matches: list[TopicTag] = [labels[i] for i in np.where(sim >= threshold)[0]]
    return matches if matches else []


def assign_topics(
    sec_item: str | None,
    chunk_embedding: np.ndarray | None,
    semantic_chunker: SemanticChunk,
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[TopicTag]:
    direct = classify_by_item(sec_item)
    if direct is not None:
        return direct
    fallback: list[TopicTag] = ["other"]
    if sec_item not in _AMBIGUOUS_ITEMS or chunk_embedding is None:
        return fallback
    matches = classify_by_embedding(chunk_embedding, semantic_chunker, threshold)
    return matches if matches else fallback
