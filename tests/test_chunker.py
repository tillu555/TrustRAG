import pytest

from src.ingestion.chunker import ScientificChunker


def test_chunker_creates_chunks():
    document = {
        "file_name": "research.pdf",
        "sections": [
            {
                "title": "Methodology",
                "page_numbers": [4, 5],
                "text": (
                    "We propose a retrieval framework. "
                    "The framework combines dense retrieval. "
                    "We also use keyword retrieval. "
                    "The final results are reranked."
                ),
            }
        ],
    }

    chunker = ScientificChunker(
        chunk_size=12,
        overlap=3,
    )

    chunks = chunker.chunk_document(document)

    assert len(chunks) > 1
    assert chunks[0]["section"] == "Methodology"
    assert chunks[0]["document_name"] == "research.pdf"
    assert chunks[0]["page_numbers"] == [4, 5]
    assert chunks[0]["word_count"] > 0


def test_chunker_handles_empty_text():
    document = {
        "file_name": "empty.pdf",
        "sections": [
            {
                "title": "Introduction",
                "page_numbers": [],
                "text": "",
            }
        ],
    }

    chunks = ScientificChunker().chunk_document(document)

    assert chunks == []


def test_chunker_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        ScientificChunker(chunk_size=0)

    with pytest.raises(ValueError):
        ScientificChunker(chunk_size=100, overlap=100)


def test_chunk_ids_are_unique():
    document = {
        "file_name": "paper.pdf",
        "sections": [
            {
                "title": "Introduction",
                "page_numbers": [1],
                "text": (
                    "First scientific sentence. "
                    "Second scientific sentence. "
                    "Third scientific sentence. "
                    "Fourth scientific sentence."
                ),
            }
        ],
    }

    chunks = ScientificChunker(
        chunk_size=5,
        overlap=1,
    ).chunk_document(document)

    ids = [chunk["chunk_id"] for chunk in chunks]

    assert len(ids) == len(set(ids))


def test_chunker_preserves_exact_page_numbers():
    document = {
        "file_name": "pages.pdf",
        "sections": [
            {
                "title": "Introduction",
                "text": (
                    "First page contains the introduction. "
                    "Second page explains the research problem."
                ),
                "page_numbers": [1, 2],
                "page_texts": [
                    {
                        "page_number": 1,
                        "text": "First page contains the introduction.",
                    },
                    {
                        "page_number": 2,
                        "text": "Second page explains the research problem.",
                    },
                ],
            }
        ],
    }

    chunks = ScientificChunker(
        chunk_size=7,
        overlap=0,
    ).chunk_document(document)

    assert len(chunks) == 2
    assert chunks[0]["page_numbers"] == [1]
    assert chunks[1]["page_numbers"] == [2]
