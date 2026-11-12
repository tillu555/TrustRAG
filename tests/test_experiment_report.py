import json

from src.evaluation.experiment_report import generate_experiment_report


def test_experiment_report_generation(tmp_path, monkeypatch):
    import src.evaluation.experiment_report as module

    monkeypatch.setattr(module, "BASE", tmp_path)

    retrieval = {
        "query_count": 2,
        "summary": {
            "dense": {"precision@5": 0.6, "mrr": 0.7},
            "hybrid_reranked": {"precision@5": 0.8, "mrr": 0.9}
        }
    }

    (tmp_path / "retrieval_report.json").write_text(
        json.dumps(retrieval), encoding="utf-8"
    )

    generate_experiment_report()

    report = json.loads(
        (tmp_path / "experiment_report.json").read_text(encoding="utf-8")
    )

    assert report["project"] == "TrustRAG"
    assert report["retrieval_evaluation"]["query_count"] == 2
    assert (tmp_path / "retrieval_comparison.csv").exists()
