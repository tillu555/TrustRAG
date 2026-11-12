import numpy as np
import pytest

from src.retrieval.embedding import EmbeddingEngine


@pytest.fixture(scope="module")
def engine():
    return EmbeddingEngine(batch_size=2)


def test_embedding_dimension(engine):
    chunks = [
        {"text": "Retrieval augmented generation improves factual grounding."},
        {"text": "Semantic search finds relevant scientific documents."},
    ]

    embeddings = engine.embed_documents(chunks)

    assert embeddings.shape == (2, 384)
    assert embeddings.dtype == np.float32


def test_embeddings_are_normalized(engine):
    chunks = [
        {"text": "Scientific document retrieval."},
        {"text": "Evidence verification in artificial intelligence."},
    ]

    embeddings = engine.embed_documents(chunks)

    norms = np.linalg.norm(embeddings, axis=1)

    assert np.allclose(norms, 1.0, atol=1e-5)


def test_query_embedding(engine):
    embedding = engine.embed_query(
        "How does retrieval improve AI reliability?"
    )

    assert embedding.shape == (384,)
    assert np.isclose(np.linalg.norm(embedding), 1.0, atol=1e-5)


def test_empty_query_rejected(engine):
    with pytest.raises(ValueError):
        engine.embed_query("")


def test_empty_chunks(engine):
    embeddings = engine.embed_documents([])

    assert embeddings.shape == (0, 384)


def test_invalid_batch_size():
    with pytest.raises(ValueError):
        EmbeddingEngine(batch_size=0)
