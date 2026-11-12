import json
from pathlib import Path
from datetime import datetime, timezone

from src.api.app import get_pipeline

BASE = Path("data/evaluation")
BENCHMARK = BASE / "benchmark.json"
OUTPUT = BASE / "answer_review_template.json"

if not BENCHMARK.exists():
    raise SystemExit("ERROR: Create and save benchmark.json first.")

benchmark = json.loads(BENCHMARK.read_text(encoding="utf-8"))

if not benchmark:
    raise SystemExit("ERROR: Benchmark is empty.")

pipeline = get_pipeline()

if not getattr(pipeline, "documents", None):
    raise SystemExit("ERROR: No documents loaded. Start the backend and ingest your PDFs.")

reviews = []

for index, item in enumerate(benchmark, start=1):
    question = item["question"]
    print(f"[{index}/{len(benchmark)}] Generating answer...")

    try:
        result = pipeline.ask(question)
        if not isinstance(result, dict):
            result = {"answer": str(result)}

        reviews.append({
            "question": question,
            "reference_chunk_ids": item["relevant_chunk_ids"],
            "generated_answer": result.get("answer", ""),
            "retrieved_evidence": result.get("retrieved_evidence", []),
            "sources": result.get("sources", []),
            "verification_status": result.get("verification_status", "unknown"),
            "verification": result.get("verification", {}),
            "human_review": {
                "faithful": None,
                "citations_correct": None,
                "unsupported_claims": None,
                "reviewer_notes": ""
            }
        })
    except Exception as exc:
        reviews.append({
            "question": question,
            "reference_chunk_ids": item["relevant_chunk_ids"],
            "generation_error": str(exc),
            "human_review": {
                "faithful": None,
                "citations_correct": None,
                "unsupported_claims": None,
                "reviewer_notes": ""
            }
        })

BASE.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(
    json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "review_status": "pending_human_review",
        "question_count": len(reviews),
        "reviews": reviews
    }, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print(f"\nCreated: {OUTPUT}")
print(f"Questions prepared: {len(reviews)}")
print("Complete the human_review fields before reporting answer-quality results.")
