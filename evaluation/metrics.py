"""Ranking metrics with binary relevance."""

import math
from typing import Iterable, List, Sequence


def recall_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return float("nan")
    return len(rel & set(ranked[:k])) / len(rel)


def hit_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """1 if any relevant doc is in the top-k (the 'at least one good doc reached the LLM' view)."""
    rel = set(relevant)
    if not rel:
        return float("nan")
    return 1.0 if rel & set(ranked[:k]) else 0.0


def mrr(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    rel = set(relevant)
    if not rel:
        return float("nan")
    for i, cid in enumerate(ranked, start=1):
        if cid in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return float("nan")
    dcg = sum(1.0 / math.log2(i + 1) for i, cid in enumerate(ranked[:k], start=1) if cid in rel)
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(rel), k) + 1))
    return dcg / ideal if ideal else 0.0


def mean(values: List[float]) -> float:
    vals = [v for v in values if v == v]  # drop NaN
    return sum(vals) / len(vals) if vals else float("nan")
