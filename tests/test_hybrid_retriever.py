import pytest

from src.retrieval.hybrid_retriever import HybridRetriever


class MockVectorStore:

    def search(self, query_embedding, top_k=5):
        results = [
            {
                "document": {"chunk_id": "A", "text": "Semantic result A"},
                "score": 0.95,
            },
            {
                "document": {"chunk_id": "B", "text": "Semantic result B"},
                "score": 0.85,
            },
        ]

        return results[:top_k]


class MockBM25:

    def search(self, query, top_k=5):
        results = [
            {
                "document": {"chunk_id": "B", "text": "Keyword result B"},
                "score": 4.5,
            },
            {
                "document": {"chunk_id": "C", "text": "Keyword result C"},
                "score": 3.5,
            },
        ]

        return results[:top_k]


@pytest.fixture
def retriever():
    return HybridRetriever(
        vector_store=MockVectorStore(),
        bm25_retriever=MockBM25(),
    )


def test_hybrid_returns_ranked_results(retriever):
    results = retriever.search(
        query="scientific retrieval",
        query_embedding=[1.0],
        top_k=3,
    )

    assert len(results) == 3
    assert results[0]["document"]["chunk_id"] == "B"


def test_hybrid_combines_duplicate_evidence(retriever):
    results = retriever.search(
        query="scientific retrieval",
        query_embedding=[1.0],
        top_k=5,
    )

    ids = [
        result["document"]["chunk_id"]
        for result in results
    ]

    assert len(ids) == len(set(ids))


def test_hybrid_records_retrieval_sources(retriever):
    results = retriever.search(
        query="scientific retrieval",
        query_embedding=[1.0],
        top_k=3,
    )

    result_b = next(
        item for item in results
        if item["document"]["chunk_id"] == "B"
    )

    assert "dense" in result_b["sources"]
    assert "bm25" in result_b["sources"]


def test_hybrid_records_individual_ranks(retriever):
    results = retriever.search(
        query="scientific retrieval",
        query_embedding=[1.0],
        top_k=3,
    )

    result_b = next(
        item for item in results
        if item["document"]["chunk_id"] == "B"
    )

    assert result_b["dense_rank"] == 2
    assert result_b["bm25_rank"] == 1


def test_hybrid_respects_top_k(retriever):
    results = retriever.search(
        query="retrieval",
        query_embedding=[1.0],
        top_k=1,
    )

    assert len(results) == 1


def test_hybrid_rejects_empty_query(retriever):
    with pytest.raises(ValueError):
        retriever.search(
            query="",
            query_embedding=[1.0],
        )


def test_hybrid_rejects_invalid_top_k(retriever):
    with pytest.raises(ValueError):
        retriever.search(
            query="retrieval",
            query_embedding=[1.0],
            top_k=0,
        )


def test_hybrid_rejects_invalid_weights():
    with pytest.raises(ValueError):
        HybridRetriever(
            MockVectorStore(),
            MockBM25(),
            dense_weight=0,
            sparse_weight=0,
        )


def test_hybrid_supports_dense_only():
    retriever = HybridRetriever(
        MockVectorStore(),
        MockBM25(),
        sparse_weight=0,
    )

    results = retriever.search(
        query="retrieval",
        query_embedding=[1.0],
    )

    assert len(results) == 2
    assert results[0]["sources"] == ["dense"]


def test_hybrid_supports_sparse_only():
    retriever = HybridRetriever(
        MockVectorStore(),
        MockBM25(),
        dense_weight=0,
    )

    results = retriever.search(
        query="retrieval",
        query_embedding=[1.0],
    )

    assert len(results) == 2
    assert results[0]["sources"] == ["bm25"]
