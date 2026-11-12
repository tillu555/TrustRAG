
from pathlib import Path
from typing import Any

import pymupdf


class PDFLoader:
    """Extract text and metadata from scientific PDF documents."""

    def load(self, file_path: str | Path) -> dict[str, Any]:
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {path}")

        if path.suffix.lower() != ".pdf":
            raise ValueError("Only PDF files are supported.")

        pages = []

        with pymupdf.open(path) as document:
            metadata = document.metadata or {}

            if document.is_encrypted:
                raise ValueError("Encrypted PDFs are not supported.")

            for page_number, page in enumerate(document, start=1):
                text = page.get_text("text", sort=True).strip()

                pages.append({
                    "page_number": page_number,
                    "text": text,
                    "word_count": len(text.split()),
                })

            full_text = "\n\n".join(
                page["text"] for page in pages
            )

            return {
                "file_name": path.name,
                "file_path": str(path),
                "title": metadata.get("title") or path.stem,
                "author": metadata.get("author") or "Unknown",
                "page_count": len(pages),
                "word_count": len(full_text.split()),
                "pages": pages,
                "full_text": full_text,
            }


if __name__ == "__main__":
    print("TrustRAG PDF Loader is ready.")
