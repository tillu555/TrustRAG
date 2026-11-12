import numpy as np
import pytest

from src.retrieval.reranker import CrossEncoderReranker


class MockCrossEncoder:

    def predict(
        self,
        pairs,
        batch_size=8,
        show_progress_bar=False,
        convert_to_numpy=True,
    ):
        scores = []

        for query, passage in pairs:
            if "relevant" in passage.lower():
                scores.append(0.95)
            elif "partially relevant" in passage.lower():
                scores.append(0.65)
            else:
                scores.append(0.10)

        return np.array(scores, dtype=np.float32)


@pytest.fixture
def reranker():
    return CrossEncoderReranker(model=MockCrossEncoder())


@pytest.fixture
def candidates():
    return [
        {
            "document": {
                "chunk_id": "A",
                "text": "Unrelated information.",
            },
            "rrf_score": 0.05,
        },
        {
            "document": {
                "chunk_id": "B",
                "text": "Highly relevant scientific evidence.",
            },
            "rrf_score": 0.04,
        },
        {
            "document": {
                "chunk_id": "C",
                "text": "Partially relevant evidence.",
            },
            "rrf_score": 0.03,
        },
    ]


def test_reranker_orders_by_relevance(reranker, candidates):
    results = reranker.rerank(
        "scientific retrieval",
        candidates,
        top_k=3,
    )

    ids = [
        item["document"]["chunk_id"]
        for item in results
    ]

    assert ids == ["B", "C", "A"]


def test_reranker_adds_scores(reranker, candidates):
    results = reranker.rerank(
        "scientific retrieval",
        candidates,
    )

    assert all("rerank_score" in item for item in results)


def test_reranker_respects_top_k(reranker, candidates):
    results = reranker.rerank(
        "scientific retrieval",
        candidates,
        top_k=2,
    )

    assert len(results) == 2


def test_reranker_handles_empty_candidates(reranker):
    assert reranker.rerank(
        "scientific retrieval",
        [],
    ) == []


def test_reranker_rejects_empty_query(reranker, candidates):
    with pytest.raises(ValueError):
        reranker.rerank("", candidates)


def test_reranker_rejects_invalid_top_k(reranker, candidates):
    with pytest.raises(ValueError):
        reranker.rerank(
            "scientific retrieval",
            candidates,
            top_k=0,
        )


def test_reranker_rejects_missing_text(reranker):
    candidates = [
        {"document": {"chunk_id": "A"}}
    ]

    with pytest.raises(ValueError):
        reranker.rerank("retrieval", candidates)


def test_reranker_rejects_invalid_scores(candidates):
    class InvalidModel:
        def predict(self, pairs, **kwargs):
            return np.array([float("nan")] * len(pairs))

    reranker = CrossEncoderReranker(model=InvalidModel())

    with pytest.raises(ValueError):
        reranker.rerank("retrieval", candidates)


def test_reranker_preserves_original_metadata(reranker, candidates):
    results = reranker.rerank(
        "scientific retrieval",
        candidates,
    )

    best = results[0]

    assert best["document"]["chunk_id"] == "B"
    assert best["rrf_score"] == 0.04
