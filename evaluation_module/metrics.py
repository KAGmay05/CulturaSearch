"""Core retrieval metrics used by the evaluation module."""

from __future__ import annotations

import math
from typing import Sequence


def precision_at_k(relevant_hits: int, k: int) -> float:
    if k <= 0:
        return 0.0
    return float(relevant_hits) / float(k)


def recall_at_k(relevant_hits: int, total_relevant: int) -> float:
    if total_relevant <= 0:
        return 0.0
    return float(relevant_hits) / float(total_relevant)


def f1_at_k(precision: float, recall: float) -> float:
    if precision <= 0.0 or recall <= 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    score = 0.0
    for index, relevance in enumerate(relevances[:k]):
        score += (2.0**float(relevance) - 1.0) / math.log2(index + 2.0)
    return score


def ndcg_at_k(relevances: Sequence[float], k: int) -> float:
    actual_dcg = dcg_at_k(relevances, k)
    ideal_relevances = sorted(relevances, reverse=True)
    ideal_dcg = dcg_at_k(ideal_relevances, k)
    if ideal_dcg <= 0.0:
        return 0.0
    return actual_dcg / ideal_dcg


def mrr_at_k(relevances: Sequence[float], k: int) -> float:
    for index, relevance in enumerate(relevances[:k], start=1):
        if relevance > 0:
            return 1.0 / float(index)
    return 0.0