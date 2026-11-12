#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."
source .venv/Scripts/activate

echo "=== TrustRAG Research Evaluation ==="

if [ ! -f data/evaluation/annotation_candidates.json ]; then
    echo "[1/3] Preparing annotation candidates..."
    python -m src.evaluation.prepare_benchmark
else
    echo "[1/3] Annotation candidates found."
fi

if [ ! -f data/evaluation/benchmark.json ]; then
    echo "ERROR: Annotate questions in the Benchmark page and save benchmark.json first."
    exit 1
fi

echo "[2/3] Validating benchmark..."
python -m src.evaluation.validate_benchmark

echo "[3/3] Running retrieval experiments..."
python -m src.evaluation.run_experiment --k 5 --candidate-k 30


echo "[4/4] Recording experiment environment..."

python - <<'PYMETA'
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import version, PackageNotFoundError

def package_version(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"

def command_output(command):
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unavailable"

metadata = {
    "experiment_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "git_commit": command_output(["git", "rev-parse", "HEAD"]),
    "python_version": sys.version,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "processor": platform.processor(),
    "packages": {
        name: package_version(name)
        for name in [
            "torch", "sentence-transformers", "transformers",
            "faiss-cpu", "numpy", "fastapi", "scikit-learn"
        ]
    },
    "ollama_models": command_output(["ollama", "list"]),
    "evaluation": {
        "k": 5,
        "candidate_k": 30,
        "methods": ["dense", "bm25", "hybrid", "hybrid_reranked"]
    }
}

path = Path("data/evaluation/experiment_metadata.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
print(f"Saved reproducibility metadata: {path}")
PYMETA

echo "=== Evaluation complete ==="
echo "JSON: data/evaluation/retrieval_report.json"
echo "CSV:  data/evaluation/ablation_comparison.csv"
