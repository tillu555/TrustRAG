import pytest

from src.retrieval.bm25_retriever import BM25Retriever


@pytest.fixture
def documents():
    return [
        {
            "chunk_id": "chunk_001",
            "text": "BERT uses transformer architecture for language understanding.",
        },
        {
            "chunk_id": "chunk_002",
            "text": "FAISS enables efficient vector similarity search.",
        },
        {
            "chunk_id": "chunk_003",
            "text": "BM25 is a probabilistic lexical information retrieval algorithm.",
        },
        {
            "chunk_id": "chunk_004",
            "text": "Retrieval augmented generation combines retrieval with language models.",
        },
    ]


@pytest.fixture
def retriever(documents):
    engine = BM25Retriever()
    engine.fit(documents)
    return engine


def test_bm25_index_creation(retriever):
    assert len(retriever) == 4


def test_bm25_retrieves_relevant_document(retriever):
    results = retriever.search("FAISS vector similarity search")

    assert len(results) > 0
    assert results[0]["document"]["chunk_id"] == "chunk_002"


def test_bm25_returns_scores(retriever):
    results = retriever.search("transformer architecture")

    assert isinstance(results[0]["score"], float)
    assert results[0]["score"] > 0


def test_bm25_respects_top_k(retriever):
    results = retriever.search("retrieval", top_k=2)

    assert len(results) == 2


def test_bm25_handles_empty_query(retriever):
    assert retriever.search("!!!") == []


def test_bm25_rejects_empty_documents():
    engine = BM25Retriever()

    with pytest.raises(ValueError):
        engine.fit([])


def test_bm25_rejects_invalid_top_k(retriever):
    with pytest.raises(ValueError):
        retriever.search("retrieval", top_k=0)


def test_bm25_requires_fitting():
    engine = BM25Retriever()

    with pytest.raises(RuntimeError):
        engine.search("scientific retrieval")


def test_bm25_rejects_invalid_document():
    engine = BM25Retriever()

    with pytest.raises(ValueError):
        engine.fit([{"text": ""}])


def test_bm25_tokenization():
    tokens = BM25Retriever.tokenize(
        "BERT-based models improve scientific NLP."
    )

    assert "bert-based" in tokens
    assert "scientific" in tokens
