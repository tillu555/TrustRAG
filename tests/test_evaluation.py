import pytest

from src.evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    ndcg_at_k,
    hit_rate_at_k,
    evaluate_ranking,
)


def test_precision_at_k():
    assert precision_at_k(["a", "b", "c"], {"a", "c"}, 3) == pytest.approx(2 / 3)


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], {"a", "c", "d"}, 3) == pytest.approx(2 / 3)


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "b", "a"], {"a", "b"}) == 0.5


def test_reciprocal_rank_no_match():
    assert reciprocal_rank(["x", "y"], {"a"}) == 0.0


def test_ndcg_at_k():
    assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, 3) == pytest.approx(1.0)


def test_hit_rate_at_k():
    assert hit_rate_at_k(["x", "b"], {"b"}, 2) == 1.0


def test_empty_relevant_set():
    assert recall_at_k(["a"], set(), 5) == 0.0
    assert ndcg_at_k(["a"], set(), 5) == 0.0


def test_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k(["a"], {"a"}, 0)


def test_evaluate_ranking():
    result = evaluate_ranking(["a", "b", "c"], {"a", "c"}, 3)

    assert result["precision@3"] == pytest.approx(2 / 3)
    assert result["recall@3"] == 1.0
    assert result["mrr"] == 1.0
    assert result["hit_rate@3"] == 1.0


def test_citation_metrics():
    from src.evaluation.metrics import citation_metrics

    result = citation_metrics(
        ["S1", "S2", "S3"],
        ["S1", "S2"],
    )

    assert result["citation_precision"] == pytest.approx(2 / 3)
    assert result["citation_recall"] == 1.0
    assert result["citation_f1"] == pytest.approx(0.8)


def test_claim_verification_metrics():
    from src.evaluation.metrics import claim_verification_metrics

    result = claim_verification_metrics(
        ["SUPPORTED", "NOT_SUPPORTED", "SUPPORTED", "NOT_SUPPORTED"],
        ["SUPPORTED", "NOT_SUPPORTED", "NOT_SUPPORTED", "SUPPORTED"],
    )

    assert result["accuracy"] == 0.5
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5


def test_claim_verification_rejects_invalid_labels():
    from src.evaluation.metrics import claim_verification_metrics

    with pytest.raises(ValueError):
        claim_verification_metrics(["UNKNOWN"], ["SUPPORTED"])


def test_claim_verification_rejects_length_mismatch():
    from src.evaluation.metrics import claim_verification_metrics

    with pytest.raises(ValueError):
        claim_verification_metrics(["SUPPORTED"], [])


def test_citation_metrics_empty():
    from src.evaluation.metrics import citation_metrics

    result = citation_metrics([], [])
    assert result["citation_precision"] == 1.0
    assert result["citation_recall"] == 1.0
