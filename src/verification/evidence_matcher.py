from typing import Any


class EvidenceMatcher:
    """Connect retrieved passages to the source IDs used in the answer."""

    def prepare(
        self,
        evidence: list[dict[str, Any]],
        sources: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        prepared = []

        for index, item in enumerate(evidence):
            document = item.get("document", item)

            if not isinstance(document, dict):
                continue

            text = document.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue

            source = sources[index] if index < len(sources) else {}

            prepared.append({
                "source_id": source.get("source_id", f"S{index + 1}"),
                "text": text.strip(),
                "document_name": source.get(
                    "document_name",
                    document.get("document_name", "Unknown"),
                ),
                "section": source.get(
                    "section",
                    document.get("section", "Unknown"),
                ),
                "page_numbers": source.get(
                    "page_numbers",
                    document.get("page_numbers", []),
                ),
                "chunk_id": source.get(
                    "chunk_id",
                    document.get("chunk_id", "Unknown"),
                ),
            })

        return prepared
