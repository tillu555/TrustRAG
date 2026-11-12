import re
from typing import Any

from rank_bm25 import BM25Okapi


class BM25Retriever:
    """
    Sparse keyword retrieval for scientific documents.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        if k1 <= 0:
            raise ValueError("k1 must be positive.")

        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1.")

        self.k1 = k1
        self.b = b
        self.documents: list[dict[str, Any]] = []
        self.tokenized_corpus: list[list[str]] = []
        self.index = None

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """
        Tokenize text while preserving scientific terms.
        """
        return re.findall(
            r"(?u)\b[\w]+(?:[-'][\w]+)*\b",
            text.lower(),
        )

    def fit(self, documents: list[dict[str, Any]]) -> None:
        """
        Build the BM25 index from document chunks.
        """
        if not documents:
            raise ValueError("Cannot index an empty document collection.")

        if any(
            not isinstance(doc.get("text"), str) or not doc["text"].strip()
            for doc in documents
        ):
            raise ValueError("Every document must contain non-empty text.")

        tokenized_corpus = [
            self.tokenize(doc["text"])
            for doc in documents
        ]

        if any(not tokens for tokens in tokenized_corpus):
            raise ValueError("A document contains no searchable tokens.")

        self.documents = list(documents)
        self.tokenized_corpus = tokenized_corpus

        self.index = BM25Okapi(
            self.tokenized_corpus,
            k1=self.k1,
            b=self.b,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Return the most relevant documents for a query.
        """
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        if self.index is None:
            raise RuntimeError("Build the BM25 index using fit() first.")

        query_tokens = self.tokenize(query)

        if not query_tokens:
            return []

        scores = self.index.get_scores(query_tokens)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: (-float(scores[i]), i),
        )[:top_k]

        return [
            {
                "document": self.documents[index],
                "score": float(scores[index]),
            }
            for index in ranked_indices
        ]

    def __len__(self) -> int:
        return len(self.documents)
