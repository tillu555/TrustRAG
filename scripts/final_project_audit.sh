#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.."
source .venv/Scripts/activate

mkdir -p reports
REPORT="reports/final_project_audit.txt"

exec > >(tee "$REPORT") 2>&1

echo "======================================"
echo "       TRUSTRAG FINAL PROJECT AUDIT"
echo "======================================"
date

echo ""
echo "1. PYTHON COMPILATION"
python -m compileall -q src tests
echo "Python compilation completed."

echo ""
echo "2. BACKEND TESTS"
python -m pytest tests/ -q --tb=short
TEST_STATUS=$?

echo ""
echo "3. FRONTEND PRODUCTION BUILD"
(
  cd frontend
  npm run build
)
FRONTEND_STATUS=$?

echo ""
echo "4. REQUIRED PROJECT FILES"
for file in \
  README.md \
  requirements.txt \
  src/api/app.py \
  src/pipeline.py \
  src/evaluation/metrics.py \
  src/evaluation/runner.py \
  frontend/package.json
do
  if [ -f "$file" ]; then
    echo "PASS: $file"
  else
    echo "MISSING: $file"
  fi
done

echo ""
echo "5. RESEARCH DATA READINESS"

python - <<'PY'
import json
from pathlib import Path

base = Path("data/evaluation")

for name in [
    "annotation_candidates.json",
    "benchmark.json",
    "retrieval_report.json",
    "answer_review_template.json",
    "answer_quality_results.json",
    "experiment_metadata.json"
]:
    path = base / name
    print(f"{'FOUND' if path.exists() else 'PENDING'}: {name}")

benchmark = base / "benchmark.json"
if benchmark.exists():
    try:
        data = json.loads(benchmark.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("queries", data.get("questions", []))
        print("Benchmark questions:", len(data) if isinstance(data, list) else 0)
    except Exception:
        print("Benchmark format needs correction.")
PY

echo ""
echo "6. GIT STATUS"
git status --short

echo ""
echo "======================================"
echo "AUDIT SUMMARY"
echo "Backend tests exit code: $TEST_STATUS"
echo "Frontend build exit code: $FRONTEND_STATUS"
echo "Full report: $REPORT"
echo "======================================"

if [ "$TEST_STATUS" -ne 0 ] || [ "$FRONTEND_STATUS" -ne 0 ]; then
  exit 1
fi
