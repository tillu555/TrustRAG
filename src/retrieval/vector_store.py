from pathlib import Path
import json
from typing import Any

import faiss
import numpy as np


class FAISSVectorStore:
    """Exact cosine-similarity vector search using FAISS."""

    def __init__(self, dimension: int = 384):
        if dimension <= 0:
            raise ValueError("Dimension must be positive.")

        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.documents: list[dict[str, Any]] = []

    def add(
        self,
        embeddings: np.ndarray,
        documents: list[dict[str, Any]],
    ) -> None:

        embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        if embeddings.shape[1] != self.dimension:
            raise ValueError("Embedding dimension mismatch.")

        if len(embeddings) != len(documents):
            raise ValueError("Embeddings and documents must match.")

        if not np.all(np.isfinite(embeddings)):
            raise ValueError("Embeddings contain invalid values.")

        if not len(embeddings):
            return

        norms = np.linalg.norm(embeddings, axis=1)

        if not np.allclose(norms, 1.0, atol=1e-4):
            raise ValueError("Embeddings must be normalized.")

        self.index.add(np.ascontiguousarray(embeddings))
        self.documents.extend(documents)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        if query_embedding.shape != (1, self.dimension):
            raise ValueError("Invalid query embedding dimension.")

        if not np.all(np.isfinite(query_embedding)):
            raise ValueError("Query embedding contains invalid values.")

        if not np.isclose(
            np.linalg.norm(query_embedding),
            1.0,
            atol=1e-4,
        ):
            raise ValueError("Query embedding must be normalized.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        if self.index.ntotal == 0:
            return []

        scores, indices = self.index.search(
            query_embedding,
            min(top_k, self.index.ntotal),
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue

            results.append({
                "document": self.documents[int(index)],
                "score": float(score),
            })

        return results

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(
            self.index,
            str(directory / "index.faiss"),
        )

        with open(
            directory / "documents.json",
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.documents,
                file,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, directory: str | Path):
        directory = Path(directory)

        index_path = directory / "index.faiss"
        documents_path = directory / "documents.json"

        if not index_path.exists() or not documents_path.exists():
            raise FileNotFoundError("FAISS index files not found.")

        index = faiss.read_index(str(index_path))

        with open(documents_path, encoding="utf-8") as file:
            documents = json.load(file)

        if index.ntotal != len(documents):
            raise ValueError("Index and document count mismatch.")

        store = cls(dimension=index.d)
        store.index = index
        store.documents = documents

        return store

    def __len__(self) -> int:
        return self.index.ntotal
