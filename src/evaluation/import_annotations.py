import csv
import json
from pathlib import Path
from collections import defaultdict

BASE = Path("data/evaluation")
SOURCE = BASE / "annotation_worksheet.csv"
OUTPUT = BASE / "benchmark.json"

if not SOURCE.exists():
    raise SystemExit("Annotation worksheet not found.")

groups = defaultdict(set)
valid_ids = set()

candidates_path = BASE / "annotation_candidates.json"
if not candidates_path.exists():
    raise SystemExit("Annotation candidates not found.")

candidates = json.loads(candidates_path.read_text(encoding="utf-8"))

def collect(value):
    if isinstance(value, list):
        for item in value:
            collect(item)
    elif isinstance(value, dict):
        if value.get("chunk_id") is not None:
            valid_ids.add(str(value["chunk_id"]))
        for child in value.values():
            if isinstance(child, (dict, list)):
                collect(child)

collect(candidates)

with SOURCE.open(encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        question = (row.get("research_question") or "").strip()
        chunk_id = (row.get("chunk_id") or "").strip()
        relevant = (row.get("relevant") or "").strip().lower()

        if not question:
            continue
        if relevant not in {"yes", "no"}:
            continue
        if chunk_id not in valid_ids:
            raise SystemExit(f"Unknown chunk ID: {chunk_id}")

        if relevant == "yes":
            groups[question].add(chunk_id)

benchmark = [
    {
        "question": question,
        "relevant_chunk_ids": sorted(ids)
    }
    for question, ids in groups.items()
    if ids
]

if not benchmark:
    raise SystemExit(
        "No valid annotations found. Add questions and mark supporting chunks as yes."
    )

OUTPUT.write_text(
    json.dumps(benchmark, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("Benchmark created:", OUTPUT)
print("Questions:", len(benchmark))
print("Relevant chunk labels:", sum(len(x["relevant_chunk_ids"]) for x in benchmark))
