import numpy as np
import pytest

from src.pipeline import TrustRAGPipeline
from src.ingestion.section_parser import SectionParser
from src.ingestion.chunker import ScientificChunker
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.bm25_retriever import BM25Retriever


class FakePDFLoader:
    def load(self, file_path):
        return {
            "file_name": file_path.name,
            "page_count": 1,
            "pages": [
                {
                    "page_number": 1,
                    "text": (
                        "Introduction\n"
                        "Retrieval augmented generation combines "
                        "retrieval systems with language models. "
                        "It provides external evidence to support "
                        "generated answers."
                    ),
                    "word_count": 23,
                }
            ],
            "full_text": (
                "Introduction\n"
                "Retrieval augmented generation combines "
                "retrieval systems with language models. "
                "It provides external evidence to support "
                "generated answers."
            ),
        }


class FakeEmbeddingEngine:
    embedding_dimension = 3

    def embed_documents(self, chunks):
        return np.tile(
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
            (len(chunks), 1),
        )

    def embed_query(self, query):
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)


class FakeReranker:
    def rerank(self, query, candidates, top_k=5):
        results = []

        for candidate in candidates[:top_k]:
            item = dict(candidate)
            item["rerank_score"] = 1.0
            results.append(item)

        return results


class FakeGenerator:
    def generate(self, query, evidence):
        return {
            "answer": "RAG retrieves evidence before generating answers.",
            "model": "fake-test-model",
            "sources": [
                {
                    "source_id": "S1",
                    "document_name": evidence[0]["document"]["document_name"],
                    "section": evidence[0]["document"]["section"],
                    "page_numbers": evidence[0]["document"]["page_numbers"],
                }
            ],
        }


@pytest.fixture
def pipeline():
    return TrustRAGPipeline(
        pdf_loader=FakePDFLoader(),
        section_parser=SectionParser(),
        chunker=ScientificChunker(chunk_size=100, overlap=10),
        embedding_engine=FakeEmbeddingEngine(),
        vector_store=FAISSVectorStore(dimension=3),
        bm25_retriever=BM25Retriever(),
        reranker=FakeReranker(),
        generator=FakeGenerator(),
    )


def test_pipeline_ingests_pdf(pipeline, tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"fake PDF used by mocked loader")

    result = pipeline.ingest_pdf(pdf_path)

    assert result["file_name"] == "paper.pdf"
    assert result["chunk_count"] > 0
    assert result["already_ingested"] is False
    assert pipeline.get_library_summary()["document_count"] == 1


def test_pipeline_prevents_duplicate_ingestion(pipeline, tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"fake PDF")

    first = pipeline.ingest_pdf(pdf_path)
    second = pipeline.ingest_pdf(pdf_path)

    assert first["document_id"] == second["document_id"]
    assert second["already_ingested"] is True
    assert pipeline.get_library_summary()["document_count"] == 1


def test_pipeline_answers_question(pipeline, tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"fake PDF")

    pipeline.ingest_pdf(pdf_path)

    result = pipeline.ask(
        "How does retrieval augmented generation work?"
    )

    assert result["answer"]
    assert result["model"] == "fake-test-model"
    assert result["sources"]
    assert result["retrieved_evidence"]
    assert result["verification_status"] == "not_run"


def test_pipeline_rejects_question_before_ingestion(pipeline):
    with pytest.raises(RuntimeError, match="No documents"):
        pipeline.ask("What is the main contribution?")


def test_pipeline_rejects_empty_question(pipeline):
    with pytest.raises(ValueError, match="Question"):
        pipeline.ask("   ")


def test_pipeline_rejects_non_pdf(pipeline, tmp_path):
    file_path = tmp_path / "paper.txt"
    file_path.write_text("not a PDF")

    with pytest.raises(ValueError, match="PDF"):
        pipeline.ingest_pdf(file_path)


def test_pipeline_uses_unique_chunk_ids_across_documents(
    pipeline, tmp_path
):
    first_path = tmp_path / "first.pdf"
    second_path = tmp_path / "second.pdf"

    first_path.write_bytes(b"first")
    second_path.write_bytes(b"second")

    pipeline.ingest_pdf(first_path)
    pipeline.ingest_pdf(second_path)

    chunk_ids = [
        chunk["chunk_id"] for chunk in pipeline.documents
    ]

    assert len(chunk_ids) == len(set(chunk_ids))
    assert pipeline.get_library_summary()["document_count"] == 2


def test_pipeline_saves_and_restores_library(pipeline, tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"fake PDF")

    pipeline.ingest_pdf(pdf_path)

    library_path = tmp_path / "saved_library"
    pipeline.save_library(library_path)

    restored = TrustRAGPipeline(
        pdf_loader=FakePDFLoader(),
        section_parser=SectionParser(),
        chunker=ScientificChunker(chunk_size=100, overlap=10),
        embedding_engine=FakeEmbeddingEngine(),
        vector_store=FAISSVectorStore(dimension=3),
        bm25_retriever=BM25Retriever(),
        reranker=FakeReranker(),
        generator=FakeGenerator(),
    )

    restored.load_library(library_path)

    assert restored.get_library_summary() == (
        pipeline.get_library_summary()
    )

    assert restored.documents == pipeline.documents
    assert restored.ingested_files == pipeline.ingested_files


def test_restored_pipeline_answers_questions(pipeline, tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"fake PDF")

    pipeline.ingest_pdf(pdf_path)

    library_path = tmp_path / "saved_library"
    pipeline.save_library(library_path)

    restored = TrustRAGPipeline(
        pdf_loader=FakePDFLoader(),
        section_parser=SectionParser(),
        chunker=ScientificChunker(chunk_size=100, overlap=10),
        embedding_engine=FakeEmbeddingEngine(),
        vector_store=FAISSVectorStore(dimension=3),
        bm25_retriever=BM25Retriever(),
        reranker=FakeReranker(),
        generator=FakeGenerator(),
    )

    restored.load_library(library_path)

    answer = restored.ask("How does RAG work?")

    assert answer["answer"]
    assert answer["sources"]


def test_pipeline_rejects_missing_library(pipeline, tmp_path):
    with pytest.raises(FileNotFoundError):
        pipeline.load_library(tmp_path / "missing_library")
