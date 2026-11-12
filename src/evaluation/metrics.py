from __future__ import annotations

import math
from typing import Iterable


def precision_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")
    return sum(x in relevant_ids for x in ranked_ids[:k]) / k


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")
    if not relevant_ids:
        return 0.0
    return sum(x in relevant_ids for x in ranked_ids[:k]) / len(relevant_ids)


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")

    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(ranked_ids[:k], start=1)
        if chunk_id in relevant_ids
    )

    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    return dcg / idcg if idcg else 0.0


def hit_rate_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive.")
    return float(any(x in relevant_ids for x in ranked_ids[:k]))


def evaluate_ranking(
    ranked_ids: list[str],
    relevant_ids: Iterable[str],
    k: int = 5,
) -> dict[str, float]:
    relevant = set(relevant_ids)

    return {
        f"precision@{k}": precision_at_k(ranked_ids, relevant, k),
        f"recall@{k}": recall_at_k(ranked_ids, relevant, k),
        "mrr": reciprocal_rank(ranked_ids, relevant),
        f"ndcg@{k}": ndcg_at_k(ranked_ids, relevant, k),
        f"hit_rate@{k}": hit_rate_at_k(ranked_ids, relevant, k),
    }


def citation_metrics(
    cited_source_ids: Iterable[str],
    supporting_source_ids: Iterable[str],
) -> dict[str, float]:
    """Measure citation precision, recall and F1."""
    cited = set(cited_source_ids)
    supporting = set(supporting_source_ids)

    if not cited and not supporting:
        return {
            "citation_precision": 1.0,
            "citation_recall": 1.0,
            "citation_f1": 1.0,
        }

    true_positive = len(cited & supporting)

    precision = true_positive / len(cited) if cited else 0.0
    recall = true_positive / len(supporting) if supporting else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "citation_precision": precision,
        "citation_recall": recall,
        "citation_f1": f1,
    }


def claim_verification_metrics(
    predicted: Iterable[str],
    expected: Iterable[str],
) -> dict[str, float]:
    """
    Evaluate claim classifications.
    Supported = positive class.
    Labels must be SUPPORTED or NOT_SUPPORTED.
    """
    predicted = list(predicted)
    expected = list(expected)

    if len(predicted) != len(expected):
        raise ValueError("Predicted and expected lengths must match.")

    if not expected:
        raise ValueError("At least one labelled claim is required.")

    valid = {"SUPPORTED", "NOT_SUPPORTED"}

    if any(x not in valid for x in predicted + expected):
        raise ValueError(f"Labels must be one of {valid}.")

    tp = sum(p == e == "SUPPORTED" for p, e in zip(predicted, expected))
    tn = sum(p == e == "NOT_SUPPORTED" for p, e in zip(predicted, expected))
    fp = sum(p == "SUPPORTED" and e == "NOT_SUPPORTED" for p, e in zip(predicted, expected))
    fn = sum(p == "NOT_SUPPORTED" and e == "SUPPORTED" for p, e in zip(predicted, expected))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )
    accuracy = (tp + tn) / len(expected)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }
