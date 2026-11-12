from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def find_documents_file() -> Path:
    preferred = [
        Path("data/library/documents.json"),
        Path("data/processed/documents.json"),
        Path("data/documents.json"),
    ]

    for path in preferred:
        if path.exists():
            return path

    candidates = [
        path for path in Path("data").rglob("documents.json")
        if "evaluation" not in path.parts
        and "uploads" not in path.parts
    ]

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            "Could not find documents.json under data/. "
            "Export the indexed chunks from your saved library first."
        )

    raise RuntimeError(
        "Multiple documents.json files found. "
        "Set the correct source path explicitly."
    )


def extract_chunks(data: Any) -> list[dict]:
    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("documents", "chunks", "data"):
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError("Unrecognized documents.json structure.")


def prepare_benchmark(source: Path, output: Path) -> int:
    with source.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    chunks = extract_chunks(raw)
    candidates = []

    for item in chunks:
        if not isinstance(item, dict):
            continue

        chunk_id = item.get("chunk_id")
        text = item.get("text")

        if chunk_id is None or not isinstance(text, str) or not text.strip():
            continue

        candidates.append({
            "chunk_id": str(chunk_id),
            "document_id": item.get("document_id", ""),
            "document_name": item.get(
                "document_name",
                item.get("file_name", "Unknown")
            ),
            "section": item.get("section", "Unknown"),
            "page_numbers": item.get("page_numbers", []),
            "text": text.strip(),
        })

    if not candidates:
        raise ValueError("No valid chunks found in the source file.")

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as file:
        json.dump({
            "description": "TrustRAG manually annotated evidence candidates",
            "annotation_instructions": (
                "Create research questions and assign relevant_chunk_ids "
                "using the chunk IDs below. Only label chunks that genuinely "
                "contain evidence relevant to the question."
            ),
            "queries": [],
            "chunks": candidates,
        }, file, indent=2, ensure_ascii=False)

    return len(candidates)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/evaluation/annotation_candidates.json")
    )
    args = parser.parse_args()

    source = args.source or find_documents_file()
    count = prepare_benchmark(source, args.output)

    print(f"Source: {source}")
    print(f"Exported chunks: {count}")
    print(f"Annotation file: {args.output}")
