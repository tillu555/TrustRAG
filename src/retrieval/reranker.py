from typing import Any

import numpy as np
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Rerank retrieved scientific passages using
    query-passage relevance scoring.
    """

    MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str | None = None,
        batch_size: int = 8,
        max_length: int = 512,
        model: Any = None,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        if max_length <= 0:
            raise ValueError("max_length must be positive.")

        self.model_name = model_name
        self.batch_size = batch_size

        self.model = model or CrossEncoder(
            model_name,
            device=device,
            max_length=max_length,
        )

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Score and reorder candidate passages.

        Each candidate must contain:
        {
            "document": {"text": "..."},
            "rrf_score": float
        }
        """

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        if not candidates:
            return []

        pairs = []

        for candidate in candidates:
            document = candidate.get("document")

            if not isinstance(document, dict):
                raise ValueError("Candidate must contain a document.")

            text = document.get("text")

            if not isinstance(text, str) or not text.strip():
                raise ValueError("Candidate document text cannot be empty.")

            pairs.append((query, text))

        scores = self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        scores = np.asarray(scores)

        if scores.ndim == 2 and scores.shape[1] == 1:
            scores = scores[:, 0]

        if scores.ndim != 1 or len(scores) != len(candidates):
            raise ValueError("Unexpected Cross-Encoder score dimensions.")

        if not np.all(np.isfinite(scores)):
            raise ValueError("Reranker returned invalid scores.")

        ranked = []

        for position, (candidate, score) in enumerate(
            zip(candidates, scores)
        ):
            result = dict(candidate)
            result["rerank_score"] = float(score)
            ranked.append((position, result))

        ranked.sort(
            key=lambda item: (-item[1]["rerank_score"], item[0])
        )

        return [
            result
            for _, result in ranked[:top_k]
        ]
