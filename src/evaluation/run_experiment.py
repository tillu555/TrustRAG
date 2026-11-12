from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.api.app import get_pipeline
from src.evaluation.runner import (
    evaluate_retrieval,
    load_dataset,
    save_report,
    save_ablation_csv,
)
from src.evaluation.validate_benchmark import validate_benchmark

def main():
    parser = argparse.ArgumentParser(
        description="Run TrustRAG retrieval ablation experiments."
    )

    parser.add_argument(
        "--benchmark",
        default="data/evaluation/benchmark.json",
    )
    parser.add_argument(
        "--candidates",
        default="data/evaluation/annotation_candidates.json",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=30)

    args = parser.parse_args()

    print("\n=== TrustRAG Research Evaluation ===")

    validation = validate_benchmark(
        args.benchmark,
        args.candidates,
    )

    if not validation["valid"]:
        print("\nBenchmark validation failed:")
        print(json.dumps(validation, indent=2))
        raise SystemExit(1)

    dataset = load_dataset(args.benchmark)

    print(f"\nValidated questions: {len(dataset)}")
    print("Loading TrustRAG pipeline...")

    pipeline = get_pipeline()

    if not getattr(pipeline, "documents", None):
        raise RuntimeError(
            "No indexed documents found in the API pipeline. "
            "Load your saved library before running evaluation."
        )

    print(f"Indexed chunks: {len(pipeline.documents)}")
    print("\nRunning retrieval experiments...")

    report = evaluate_retrieval(
        pipeline=pipeline,
        dataset=dataset,
        k=args.k,
        candidate_k=args.candidate_k,
    )

    save_report(report)
    save_ablation_csv(report)

    print("\n=== RETRIEVAL COMPARISON ===")

    for method, result in report["summary"].items():
        print(f"\n{method.upper()}")
        print(json.dumps(result, indent=2))

    print("\n=== OUTPUT FILES ===")
    print("data/evaluation/retrieval_report.json")
    print("data/evaluation/ablation_comparison.csv")
    print("\nExperiment completed.")

if __name__ == "__main__":
    main()
