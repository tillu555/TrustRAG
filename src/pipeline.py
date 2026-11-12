from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.section_parser import SectionParser
from src.ingestion.chunker import ScientificChunker
from src.retrieval.embedding import EmbeddingEngine
from src.retrieval.vector_store import FAISSVectorStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.generation.ollama_generator import OllamaGenerator
from src.verification.verification_pipeline import ClaimVerificationPipeline


class TrustRAGPipeline:
    """End-to-end PDF ingestion and evidence-grounded question answering."""

    def __init__(
        self,
        pdf_loader: Any = None,
        section_parser: Any = None,
        chunker: Any = None,
        embedding_engine: Any = None,
        vector_store: Any = None,
        bm25_retriever: Any = None,
        reranker: Any = None,
        generator: Any = None,
        verification_pipeline: Any = None,
    ):
        self.pdf_loader = pdf_loader or PDFLoader()
        self.section_parser = section_parser or SectionParser()
        self.chunker = chunker or ScientificChunker()

        self.embedding_engine = embedding_engine or EmbeddingEngine()

        dimension = self.embedding_engine.embedding_dimension
        self.vector_store = vector_store or FAISSVectorStore(
            dimension=dimension
        )

        self.bm25_retriever = bm25_retriever or BM25Retriever()

        self.hybrid_retriever = HybridRetriever(
            vector_store=self.vector_store,
            bm25_retriever=self.bm25_retriever,
        )

        self.reranker = reranker or CrossEncoderReranker()
        self.generator = generator or OllamaGenerator()
        self.verification_pipeline = verification_pipeline

        self.documents: list[dict[str, Any]] = []
        self.ingested_files: dict[str, dict[str, Any]] = {}

    def ingest_pdf(self, file_path: str | Path) -> dict[str, Any]:
        """Extract, parse, chunk, embed and index one PDF."""

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError("Only PDF files are supported.")

        resolved_path = str(path.resolve())

        if resolved_path in self.ingested_files:
            existing = dict(self.ingested_files[resolved_path])
            existing["already_ingested"] = True
            return existing

        extracted = self.pdf_loader.load(path)
        parsed = self.section_parser.parse(extracted)
        chunks = self.chunker.chunk_document(parsed)

        if not chunks:
            raise ValueError(
                "No text chunks were extracted from this PDF. "
                "The document may be scanned or empty."
            )

        # Chunk IDs must be unique across different PDFs.
        document_id = hashlib.sha256(
            resolved_path.encode("utf-8")
        ).hexdigest()[:12]

        prepared_chunks = []

        for chunk in chunks:
            item = dict(chunk)
            item["document_id"] = document_id
            item["chunk_id"] = (
                f"{document_id}:{chunk['chunk_id']}"
            )
            prepared_chunks.append(item)

        embeddings = self.embedding_engine.embed_documents(
            prepared_chunks
        )

        if len(embeddings) != len(prepared_chunks):
            raise ValueError(
                "Embedding count does not match chunk count."
            )

        # Add to the dense index and update the complete sparse corpus.
        self.vector_store.add(embeddings, prepared_chunks)

        updated_documents = self.documents + prepared_chunks
        self.bm25_retriever.fit(updated_documents)

        self.documents = updated_documents

        result = {
            "document_id": document_id,
            "file_name": extracted["file_name"],
            "page_count": extracted["page_count"],
            "section_count": parsed["section_count"],
            "chunk_count": len(prepared_chunks),
            "already_ingested": False,
        }

        self.ingested_files[resolved_path] = result
        return dict(result)

    def ask(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 30,
    ) -> dict[str, Any]:
        """Retrieve evidence, rerank it and generate an answer."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Question cannot be empty.")

        if not self.documents:
            raise RuntimeError(
                "No documents have been ingested. Upload a PDF first."
            )

        if top_k <= 0 or candidate_k <= 0:
            raise ValueError(
                "top_k and candidate_k must be positive."
            )

        query_embedding = self.embedding_engine.embed_query(
            query.strip()
        )

        candidates = self.hybrid_retriever.search(
            query=query.strip(),
            query_embedding=query_embedding,
            top_k=candidate_k,
            candidate_k=candidate_k,
        )

        if not candidates:
            return {
                "answer": (
                    "I could not retrieve relevant evidence "
                    "from the indexed documents."
                ),
                "model": None,
                "sources": [],
                "retrieved_evidence": [],
                "verification_status": "not_run",
            }

        reranked = self.reranker.rerank(
            query=query.strip(),
            candidates=candidates,
            top_k=top_k,
        )

        if not reranked:
            return {
                "answer": (
                    "I could not identify sufficient relevant "
                    "evidence to answer this question."
                ),
                "model": None,
                "sources": [],
                "retrieved_evidence": [],
                "verification_status": "not_run",
            }

        generated = self.generator.generate(
            query=query.strip(),
            evidence=reranked,
        )

        result = {
            "answer": generated["answer"],
            "model": generated.get("model"),
            "sources": generated.get("sources", []),
            "retrieved_evidence": reranked,
            "verification_status": "not_run",
            "verification": None,
        }

        if self.verification_pipeline is not None:
            try:
                verification = self.verification_pipeline.verify(
                    answer=generated["answer"],
                    evidence=reranked,
                    sources=generated.get("sources", []),
                )
                result["verification_status"] = verification[
                    "verification_status"
                ]
                result["verification"] = verification
            except Exception as error:
                result["verification_status"] = "error"
                result["verification_error"] = str(error)

        return result

    def save_library(self, directory: str | Path) -> None:
        """Persist the FAISS index, chunks and ingestion registry."""

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        index_directory = directory / "faiss"
        self.vector_store.save(index_directory)

        metadata = {
            "schema_version": 1,
            "embedding_model": getattr(
                self.embedding_engine, "model_name", None
            ),
            "embedding_dimension": self.vector_store.dimension,
            "ingested_files": self.ingested_files,
            "document_count": len(self.ingested_files),
            "chunk_count": len(self.documents),
        }

        temporary_file = directory / "library.json.tmp"
        metadata_file = directory / "library.json"

        with temporary_file.open("w", encoding="utf-8") as file:
            json.dump(metadata, file, ensure_ascii=False, indent=2)

        os.replace(temporary_file, metadata_file)

    def load_library(self, directory: str | Path) -> None:
        """Restore a saved library and rebuild its BM25 index."""

        directory = Path(directory)
        metadata_file = directory / "library.json"

        if not metadata_file.exists():
            raise FileNotFoundError(
                f"Library metadata not found: {metadata_file}"
            )

        with metadata_file.open(encoding="utf-8") as file:
            metadata = json.load(file)

        if metadata.get("schema_version") != 1:
            raise ValueError("Unsupported library schema version.")

        loaded_store = FAISSVectorStore.load(
            directory / "faiss"
        )

        expected_dimension = self.embedding_engine.embedding_dimension

        if loaded_store.dimension != expected_dimension:
            raise ValueError(
                "Saved index dimension does not match embedding model."
            )

        loaded_documents = loaded_store.documents

        if len(loaded_documents) != metadata.get("chunk_count"):
            raise ValueError(
                "Saved chunk count does not match FAISS metadata."
            )

        loaded_files = metadata.get("ingested_files")

        if not isinstance(loaded_files, dict):
            raise ValueError("Invalid ingestion registry.")

        self.vector_store = loaded_store
        self.documents = loaded_documents
        self.ingested_files = loaded_files

        self.bm25_retriever.fit(self.documents)

        self.hybrid_retriever = HybridRetriever(
            vector_store=self.vector_store,
            bm25_retriever=self.bm25_retriever,
        )

    def get_library_summary(self) -> dict[str, int]:
        """Return basic statistics for the in-memory research library."""

        return {
            "document_count": len(self.ingested_files),
            "chunk_count": len(self.documents),
            "indexed_vector_count": len(self.vector_store),
        }
