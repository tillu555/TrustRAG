import json
from pathlib import Path
from datetime import datetime, timezone

BASE = Path("data/evaluation")
INPUT = BASE / "answer_review_template.json"
OUTPUT = BASE / "answer_quality_results.json"
CSV = BASE / "answer_quality_results.csv"

if not INPUT.exists():
    raise SystemExit("Run prepare_answer_review first.")

data = json.loads(INPUT.read_text(encoding="utf-8"))
reviews = data.get("reviews", [])

if not reviews:
    raise SystemExit("No answer reviews found.")

def boolean_metric(field):
    values = [
        item["human_review"][field]
        for item in reviews
        if isinstance(item.get("human_review", {}).get(field), bool)
    ]
    positive = sum(values)
    return {
        "evaluated": len(values),
        "positive": positive,
        "negative": len(values) - positive,
        "rate": round(positive / len(values), 4) if values else None
    }

faithfulness = boolean_metric("faithful")
citation_correctness = boolean_metric("citations_correct")

unsupported = [
    item["human_review"]["unsupported_claims"]
    for item in reviews
    if isinstance(item.get("human_review", {}).get("unsupported_claims"), bool)
]

unsupported_rate = (
    round(sum(unsupported) / len(unsupported), 4)
    if unsupported else None
)

predicted = []
expected = []

for item in reviews:
    human = item.get("human_review", {})
    expected_label = human.get("expected_verification")
    predicted_label = item.get("verification_status")

    if isinstance(expected_label, str) and expected_label.strip():
        if isinstance(predicted_label, str) and predicted_label.strip():
            expected.append(expected_label.strip().lower())
            predicted.append(predicted_label.strip().lower())

verification_accuracy = (
    round(
        sum(a == b for a, b in zip(predicted, expected)) / len(expected),
        4
    )
    if expected else None
)

results = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "review_status": "completed" if all(
        all(isinstance(item.get("human_review", {}).get(field), bool)
            for field in ["faithful", "citations_correct", "unsupported_claims"])
        for item in reviews
    ) else "partially_reviewed",
    "total_questions": len(reviews),
    "metrics": {
        "faithfulness": faithfulness,
        "citation_correctness": citation_correctness,
        "unsupported_claim_rate": {
            "evaluated": len(unsupported),
            "positive": sum(unsupported),
            "rate": unsupported_rate
        },
        "claim_verification_accuracy": {
            "evaluated": len(expected),
            "correct": sum(a == b for a, b in zip(predicted, expected)),
            "accuracy": verification_accuracy
        }
    },
    "notes": [
        "Metrics use only manually completed review labels.",
        "Unsupported claim rate is the fraction of reviewed answers marked as containing unsupported claims.",
        "Verification accuracy requires expected_verification labels in the human review."
    ]
}

BASE.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(results, indent=2), encoding="utf-8")

import csv
with CSV.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "evaluated", "positive_or_correct", "rate"])
    writer.writerow(["faithfulness", faithfulness["evaluated"], faithfulness["positive"], faithfulness["rate"]])
    writer.writerow(["citation_correctness", citation_correctness["evaluated"], citation_correctness["positive"], citation_correctness["rate"]])
    writer.writerow(["unsupported_claim_rate", len(unsupported), sum(unsupported), unsupported_rate])
    writer.writerow(["claim_verification_accuracy", len(expected), sum(a == b for a, b in zip(predicted, expected)), verification_accuracy])

print(json.dumps(results, indent=2))
print(f"\nSaved JSON: {OUTPUT}")
print(f"Saved CSV:  {CSV}")
