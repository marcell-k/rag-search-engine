def precision(retrieved: list[str], relevant: list[str]) -> float:
    """Fraction of retrieved items that are relevant."""
    if not retrieved:
        return 0.0
    relevant_set = set(relevant)
    hits = sum(1 for title in retrieved if title in relevant_set)
    return hits / len(retrieved)


def recall(retrieved: list[str], relevant: list[str]) -> float:
    """Fraction of relevant items that were retrieved."""
    if not relevant:
        return 0.0
    relevant_set = set(relevant)
    hits = sum(1 for title in retrieved if title in relevant_set)
    return hits / len(relevant)


def f1_score(retrieved: list[str], relevant: list[str]) -> float:
    """Harmonic mean of precision and recall."""
    p = precision(retrieved, relevant)
    r = recall(retrieved, relevant)
    if p + r == 0:
        return 0.0
    return 2 * (p * r) / (p + r)
