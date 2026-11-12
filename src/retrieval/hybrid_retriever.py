import hashlib
import json
from typing import Any


class HybridRetriever:
    """
    Combine dense and sparse retrieval using
    Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        vector_store: Any,
        bm25_retriever: Any,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
    ):
        if rrf_k <= 0:
            raise ValueError("rrf_k must be positive.")

        if dense_weight < 0 or sparse_weight < 0:
            raise ValueError("Retrieval weights cannot be negative.")

        if dense_weight + sparse_weight == 0:
            raise ValueError("At least one weight must be positive.")

        self.vector_store = vector_store
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

    @staticmethod
    def _document_key(document: dict[str, Any]) -> str:
        """
        Generate a stable identity for evidence deduplication.
        """
        chunk_id = document.get("chunk_id")

        if chunk_id is not None:
            return str(chunk_id)

        serialized = json.dumps(
            document,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )

        return hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()

    def search(
        self,
        query: str,
        query_embedding: Any,
        top_k: int = 5,
        candidate_k: int = 30,
    ) -> list[dict[str, Any]]:

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0 or candidate_k <= 0:
            raise ValueError("top_k and candidate_k must be positive.")

        dense_results = (
            self.vector_store.search(
                query_embedding,
                top_k=candidate_k,
            )
            if self.dense_weight > 0
            else []
        )

        sparse_results = (
            self.bm25_retriever.search(
                query,
                top_k=candidate_k,
            )
            if self.sparse_weight > 0
            else []
        )

        combined = {}

        def add_results(results, source, weight):
            for rank, result in enumerate(results, start=1):
                document = result["document"]
                key = self._document_key(document)

                if key not in combined:
                    combined[key] = {
                        "document": document,
                        "rrf_score": 0.0,
                        "dense_score": None,
                        "bm25_score": None,
                        "dense_rank": None,
                        "bm25_rank": None,
                        "sources": [],
                    }

                entry = combined[key]

                entry["rrf_score"] += weight / (
                    self.rrf_k + rank
                )

                entry[f"{source}_score"] = float(result["score"])
                entry[f"{source}_rank"] = rank

                if source not in entry["sources"]:
                    entry["sources"].append(source)

        add_results(
            dense_results,
            "dense",
            self.dense_weight,
        )

        add_results(
            sparse_results,
            "bm25",
            self.sparse_weight,
        )

        ranked_results = sorted(
            combined.values(),
            key=lambda item: item["rrf_score"],
            reverse=True,
        )

        return ranked_results[:top_k]
