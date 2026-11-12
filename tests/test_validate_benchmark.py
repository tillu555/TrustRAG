import json

from src.evaluation.validate_benchmark import validate_benchmark


def test_valid_benchmark(tmp_path):
    benchmark = {
        "queries": [
            {
                "question": "What is the research objective?",
                "relevant_chunk_ids": ["doc1:chunk_000001"]
            }
        ]
    }

    candidates = {
        "chunks": [
            {"chunk_id": "doc1:chunk_000001"}
        ]
    }

    bp = tmp_path / "benchmark.json"
    cp = tmp_path / "candidates.json"

    bp.write_text(json.dumps(benchmark), encoding="utf-8")
    cp.write_text(json.dumps(candidates), encoding="utf-8")

    result = validate_benchmark(bp, cp)

    assert result["valid"] is True
    assert result["query_count"] == 1


def test_unknown_chunk_id(tmp_path):
    bp = tmp_path / "benchmark.json"
    cp = tmp_path / "candidates.json"

    bp.write_text(json.dumps({
        "queries": [{
            "question": "Research question?",
            "relevant_chunk_ids": ["missing"]
        }]
    }), encoding="utf-8")

    cp.write_text(json.dumps({
        "chunks": [{"chunk_id": "existing"}]
    }), encoding="utf-8")

    result = validate_benchmark(bp, cp)

    assert result["valid"] is False
    assert any("unknown chunk ID" in error for error in result["errors"])


def test_empty_benchmark_is_invalid(tmp_path):
    bp = tmp_path / "benchmark.json"
    cp = tmp_path / "candidates.json"

    bp.write_text('{"queries": []}', encoding="utf-8")
    cp.write_text('{"chunks": []}', encoding="utf-8")

    result = validate_benchmark(bp, cp)

    assert result["valid"] is False
    assert result["query_count"] == 0
