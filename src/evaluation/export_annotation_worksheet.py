import csv
import json
from pathlib import Path

source = Path("data/evaluation/annotation_candidates.json")
output = Path("data/evaluation/annotation_worksheet.csv")

if not source.exists():
    raise SystemExit("Run python -m src.evaluation.prepare_benchmark first.")

data = json.loads(source.read_text(encoding="utf-8"))
chunks = []
seen = set()

def visit(value):
    if isinstance(value, list):
        for item in value:
            visit(item)
    elif isinstance(value, dict):
        chunk_id = value.get("chunk_id")
        content = value.get("text", value.get("content", value.get("chunk_text")))

        if chunk_id is not None and isinstance(content, str):
            chunk_id = str(chunk_id)
            if chunk_id not in seen:
                seen.add(chunk_id)
                chunks.append({
                    "chunk_id": chunk_id,
                    "document": value.get("document", value.get("document_name", "")),
                    "section": value.get("section", ""),
                    "pages": value.get("pages", value.get("page", "")),
                    "text": content
                })
                return

        for child in value.values():
            if isinstance(child, (dict, list)):
                visit(child)

visit(data)

if not chunks:
    raise SystemExit("No chunks found. Check annotation_candidates.json structure.")

output.parent.mkdir(parents=True, exist_ok=True)

with output.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "chunk_id", "document", "section", "pages",
            "text", "research_question", "relevant"
        ]
    )
    writer.writeheader()
    for chunk in chunks:
        writer.writerow({
            **chunk,
            "research_question": "",
            "relevant": ""
        })

print(f"Exported {len(chunks)} unique chunks.")
print(f"Worksheet: {output}")
