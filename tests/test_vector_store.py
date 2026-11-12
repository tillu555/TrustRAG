import numpy as np
import pytest

from src.retrieval.vector_store import FAISSVectorStore


def test_add_and_search():
    store = FAISSVectorStore(dimension=3)

    embeddings = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)

    documents = [
        {"text": "Retrieval"},
        {"text": "Verification"},
        {"text": "Generation"},
    ]

    store.add(embeddings, documents)

    results = store.search(
        np.array([1.0, 0.0, 0.0]),
        top_k=2,
    )

    assert len(results) == 2
    assert results[0]["document"]["text"] == "Retrieval"
    assert results[0]["score"] == pytest.approx(1.0)


def test_empty_store():
    store = FAISSVectorStore(dimension=3)

    assert store.search(
        np.array([1.0, 0.0, 0.0]),
    ) == []


def test_save_and_load(tmp_path):
    store = FAISSVectorStore(dimension=3)

    store.add(
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        [{"text": "Scientific evidence"}],
    )

    store.save(tmp_path)

    restored = FAISSVectorStore.load(tmp_path)

    assert len(restored) == 1

    results = restored.search(
        np.array([1.0, 0.0, 0.0])
    )

    assert results[0]["document"]["text"] == "Scientific evidence"


def test_reject_invalid_dimension():
    store = FAISSVectorStore(dimension=3)

    with pytest.raises(ValueError):
        store.add(
            np.array([[1.0, 0.0]], dtype=np.float32),
            [{"text": "Invalid"}],
        )


def test_reject_unnormalized_vectors():
    store = FAISSVectorStore(dimension=3)

    with pytest.raises(ValueError):
        store.add(
            np.array([[2.0, 0.0, 0.0]], dtype=np.float32),
            [{"text": "Invalid"}],
        )


def test_reject_mismatched_documents():
    store = FAISSVectorStore(dimension=3)

    with pytest.raises(ValueError):
        store.add(
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            [],
        )
