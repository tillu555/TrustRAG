from __future__ import annotations

import json
from pathlib import Path


def validate_benchmark(
    benchmark_path: str | Path,
    candidates_path: str | Path,
) -> dict:
    benchmark_path = Path(benchmark_path)
    candidates_path = Path(candidates_path)

    with benchmark_path.open("r", encoding="utf-8") as f:
        benchmark = json.load(f)

    with candidates_path.open("r", encoding="utf-8") as f:
        candidates = json.load(f)

    queries = benchmark.get("queries", [])
    chunks = candidates.get("chunks", [])

    if not isinstance(queries, list):
        raise ValueError("Benchmark queries must be a list.")

    valid_ids = {
        str(chunk["chunk_id"])
        for chunk in chunks
        if isinstance(chunk, dict) and "chunk_id" in chunk
    }

    errors = []
    seen_questions = set()

    for index, record in enumerate(queries):
        label = f"Query {index + 1}"

        if not isinstance(record, dict):
            errors.append(f"{label}: record must be an object.")
            continue

        question = record.get("question")
        relevant = record.get("relevant_chunk_ids")

        if not isinstance(question, str) or not question.strip():
            errors.append(f"{label}: missing question.")
            continue

        normalized = question.strip().casefold()

        if normalized in seen_questions:
            errors.append(f"{label}: duplicate question.")
        seen_questions.add(normalized)

        if not isinstance(relevant, list) or not relevant:
            errors.append(f"{label}: no relevant chunk IDs.")
            continue

        if len(relevant) != len(set(map(str, relevant))):
            errors.append(f"{label}: duplicate relevant chunk IDs.")

        for chunk_id in relevant:
            if str(chunk_id) not in valid_ids:
                errors.append(
                    f"{label}: unknown chunk ID {chunk_id}."
                )

    return {
        "valid": not errors and len(queries) > 0,
        "query_count": len(queries),
        "candidate_chunk_count": len(valid_ids),
        "errors": errors,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        default="data/evaluation/benchmark.json"
    )
    parser.add_argument(
        "--candidates",
        default="data/evaluation/annotation_candidates.json"
    )
    args = parser.parse_args()

    result = validate_benchmark(args.benchmark, args.candidates)

    print(json.dumps(result, indent=2))

    if not result["valid"]:
        raise SystemExit(1)

    print("Benchmark validation successful.")
