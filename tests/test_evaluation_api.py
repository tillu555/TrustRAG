import json
from fastapi.testclient import TestClient

from src.api.app import app


def test_evaluation_report_endpoint(monkeypatch, tmp_path):
    import pathlib

    report = {
        "evaluation": "TrustRAG Retrieval Ablation Study",
        "query_count": 2,
        "k": 5,
        "candidate_k": 30,
        "summary": {
            "dense": {
                "retrieval_metrics": {"mrr": 0.5},
                "latency": {
                    "mean_ms": 10,
                    "median_ms": 10,
                    "std_ms": 0,
                    "min_ms": 10,
                    "max_ms": 10
                },
                "queries_per_second": 100
            }
        }
    }

    report_file = tmp_path / "retrieval_report.json"
    report_file.write_text(json.dumps(report), encoding="utf-8")

    original_exists = pathlib.Path.exists
    original_open = pathlib.Path.open

    def patched_exists(self):
        if self.as_posix() == "data/evaluation/retrieval_report.json":
            return report_file.exists()
        return original_exists(self)

    def patched_open(self, *args, **kwargs):
        if self.as_posix() == "data/evaluation/retrieval_report.json":
            return report_file.open(*args, **kwargs)
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(pathlib.Path, "exists", patched_exists)
    monkeypatch.setattr(pathlib.Path, "open", patched_open)

    response = TestClient(app).get("/evaluation/report")

    assert response.status_code == 200
    assert response.json()["query_count"] == 2


def test_evaluation_report_missing():
    response = TestClient(app).get("/evaluation/report")

    if response.status_code == 404:
        assert "No evaluation report" in response.json()["detail"]
    else:
        assert response.status_code == 200
