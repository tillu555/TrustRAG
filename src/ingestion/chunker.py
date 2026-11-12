
import re
from typing import Any


class ScientificChunker:
    """Split scientific sections into overlapping, page-aware chunks."""

    def __init__(
        self,
        chunk_size: int = 350,
        overlap: int = 60,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")

        if overlap < 0 or overlap >= chunk_size:
            raise ValueError(
                "overlap must be non-negative and smaller than chunk_size."
            )

        self.chunk_size = chunk_size
        self.overlap = overlap

    def _split_sentences(self, text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            return []

        return [
            sentence.strip()
            for sentence in re.split(
                r"(?<=[.!?])\s+(?=[A-Z0-9])",
                text,
            )
            if sentence.strip()
        ]

    def _build_chunks(
        self,
        text: str,
        page_numbers: list[int],
    ) -> list[dict[str, Any]]:
        """Legacy fallback when page-level text is unavailable."""
        page_texts = [
            {"page_number": page, "text": text}
            for page in page_numbers[:1]
        ]

        if not page_texts:
            return []

        chunks = self._build_page_aware_chunks(page_texts)

        # Preserve legacy metadata for old callers that only provide
        # section-level page numbers.
        for chunk in chunks:
            chunk["page_numbers"] = sorted(set(page_numbers))

        return chunks

    def _build_page_aware_chunks(
        self,
        page_texts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        # Each sentence retains the page on which it was extracted.
        sentences: list[tuple[str, int]] = []

        for page in page_texts:
            page_number = page["page_number"]

            for sentence in self._split_sentences(page["text"]):
                sentences.append((sentence, page_number))

        if not sentences:
            return []

        chunks = []
        current_sentences: list[tuple[str, int]] = []
        current_words = 0

        for sentence, page_number in sentences:
            sentence_words = len(sentence.split())

            if (
                current_sentences
                and current_words + sentence_words > self.chunk_size
            ):
                chunks.append(self._make_chunk(current_sentences))

                overlap_sentences = []
                overlap_words = 0

                for previous in reversed(current_sentences):
                    count = len(previous[0].split())

                    if overlap_words + count > self.overlap:
                        break

                    overlap_sentences.insert(0, previous)
                    overlap_words += count

                current_sentences = overlap_sentences
                current_words = overlap_words

            current_sentences.append((sentence, page_number))
            current_words += sentence_words

        if current_sentences:
            chunks.append(self._make_chunk(current_sentences))

        return chunks

    @staticmethod
    def _make_chunk(
        sentences: list[tuple[str, int]],
    ) -> dict[str, Any]:
        return {
            "text": " ".join(sentence for sentence, _ in sentences),
            "page_numbers": sorted({
                page_number for _, page_number in sentences
            }),
        }

    def chunk_document(
        self,
        parsed_document: dict[str, Any],
    ) -> list[dict[str, Any]]:
        all_chunks = []
        chunk_id = 0

        for section in parsed_document["sections"]:
            if "page_texts" in section:
                section_chunks = self._build_page_aware_chunks(
                    section["page_texts"]
                )
            else:
                section_chunks = self._build_chunks(
                    section["text"],
                    section["page_numbers"],
                )

            for chunk in section_chunks:
                all_chunks.append({
                    "chunk_id": f"chunk_{chunk_id:06d}",
                    "document_name": parsed_document["file_name"],
                    "section": section["title"],
                    "page_numbers": chunk["page_numbers"],
                    "text": chunk["text"],
                    "word_count": len(chunk["text"].split()),
                })

                chunk_id += 1

        return all_chunks


if __name__ == "__main__":
    print("TrustRAG Scientific Chunker is ready.")
