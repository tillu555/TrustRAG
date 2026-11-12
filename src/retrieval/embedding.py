from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingEngine:
    """
    Generate normalized semantic embeddings
    for scientific document chunks.
    """

    MODEL_NAME = "BAAI/bge-small-en-v1.5"
    EMBEDDING_DIMENSION = 384

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str | None = None,
        batch_size: int = 16,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.model_name = model_name
        self.batch_size = batch_size

        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

        self.embedding_dimension = (
            self.model.get_embedding_dimension()
        )

    def embed_documents(
        self,
        chunks: list[dict[str, Any]],
    ) -> np.ndarray:

        if not chunks:
            return np.empty(
                (0, self.embedding_dimension),
                dtype=np.float32,
            )

        texts = [chunk["text"] for chunk in chunks]

        if any(not text.strip() for text in texts):
            raise ValueError("Empty document chunks are not allowed.")

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return np.asarray(embeddings, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return np.asarray(
            query_embedding,
            dtype=np.float32,
        )

    def get_model_info(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "device": str(self.model.device),
        }
