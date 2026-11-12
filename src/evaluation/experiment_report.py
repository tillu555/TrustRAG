from __future__ import annotations

import csv
import json
from pathlib import Path
from datetime import datetime, timezone


BASE = Path("data/evaluation")


def load_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def generate_experiment_report():
    retrieval = load_json(BASE / "retrieval_report.json")
    quality = load_json(BASE / "answer_quality_results.json")

    report = {
        "project": "TrustRAG",
        "report_type": "Research Evaluation",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "retrieval_evaluation": retrieval,
        "answer_quality_evaluation": quality,
        "limitations": [
            "Retrieval metrics require manually annotated relevant chunk IDs.",
            "Citation metrics require manually verified supporting source IDs.",
            "Claim verification metrics require manually labelled expected outcomes.",
            "Placeholder benchmark records must not be treated as experimental results.",
            "Results are not scientifically validated until evaluated on a representative dataset."
        ]
    }

    BASE.mkdir(parents=True, exist_ok=True)

    with (BASE / "experiment_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    if retrieval and retrieval.get("summary"):
        with (BASE / "retrieval_comparison.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            methods = retrieval["summary"]
            metric_names = sorted({
                metric
                for values in methods.values()
                for metric in values
            })

            writer = csv.writer(file)
            writer.writerow(["Retrieval Method", *metric_names])

            for method, metrics in methods.items():
                writer.writerow([
                    method,
                    *[metrics.get(metric, "") for metric in metric_names]
                ])

    print("TrustRAG experiment report generated.")
    print(f"JSON: {BASE / 'experiment_report.json'}")

    if retrieval and retrieval.get("summary"):
        print(f"CSV:  {BASE / 'retrieval_comparison.csv'}")
    else:
        print("Retrieval results not available yet.")

    if quality:
        print("Answer-quality results included.")
    else:
        print("Answer-quality results not available yet.")


if __name__ == "__main__":
    generate_experiment_report()
