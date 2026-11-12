from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

from src.evaluation.metrics import evaluate_ranking


def _chunk_ids(results: list[dict[str, Any]]) -> list[str]:
    ids = []

    for result in results:
        document = result.get("document", result)

        if isinstance(document, dict):
            chunk_id = document.get("chunk_id")
            if chunk_id is not None:
                ids.append(str(chunk_id))

    return ids


def _timed_call(function, *args, **kwargs):
    start = time.perf_counter()
    result = function(*args, **kwargs)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return result, elapsed_ms


def _latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": round(mean(values), 3),
        "median_ms": round(median(values), 3),
        "std_ms": round(pstdev(values), 3),
        "min_ms": round(min(values), 3),
        "max_ms": round(max(values), 3),
    }


def evaluate_retrieval(
    pipeline: Any,
    dataset: list[dict[str, Any]],
    k: int = 5,
    candidate_k: int = 30,
) -> dict[str, Any]:
    """
    Compare dense, BM25, hybrid and hybrid + reranker.

    Dataset format:
    {
        "question": "...",
        "relevant_chunk_ids": ["document:chunk_000001"]
    }

    Latency is measured separately for each retrieval stage.
    Model loading and initial warm-up are not excluded automatically.
    """

    if not dataset:
        raise ValueError("Evaluation dataset is empty.")

    if k <= 0 or candidate_k <= 0:
        raise ValueError("k and candidate_k must be positive.")

    if not getattr(pipeline, "documents", None):
        raise RuntimeError("Pipeline has no loaded documents.")

    methods = ["dense", "bm25", "hybrid", "hybrid_reranked"]
    per_method = {method: [] for method in methods}
    latencies = {method: [] for method in methods}

    for record in dataset:
        question = record.get("question")
        relevant_ids = record.get("relevant_chunk_ids")

        if not isinstance(question, str) or not question.strip():
            raise ValueError("Every record needs a non-empty question.")

        if not isinstance(relevant_ids, list) or not relevant_ids:
            raise ValueError(
                f"Missing relevant_chunk_ids for question: {question}"
            )

        query = question.strip()

        query_embedding, embedding_ms = _timed_call(
            pipeline.embedding_engine.embed_query, query
        )

        dense, dense_ms = _timed_call(
            pipeline.vector_store.search,
            query_embedding,
            top_k=candidate_k,
        )

        sparse, sparse_ms = _timed_call(
            pipeline.bm25_retriever.search,
            query,
            top_k=candidate_k,
        )

        hybrid, hybrid_ms = _timed_call(
            pipeline.hybrid_retriever.search,
            query=query,
            query_embedding=query_embedding,
            top_k=candidate_k,
            candidate_k=candidate_k,
        )

        reranked, rerank_ms = _timed_call(
            pipeline.reranker.rerank,
            query=query,
            candidates=hybrid,
            top_k=k,
        )

        rankings = {
            "dense": _chunk_ids(dense)[:k],
            "bm25": _chunk_ids(sparse)[:k],
            "hybrid": _chunk_ids(hybrid)[:k],
            "hybrid_reranked": _chunk_ids(reranked)[:k],
        }

        timings = {
            "dense": dense_ms,
            "bm25": sparse_ms,
            "hybrid": hybrid_ms,
            "hybrid_reranked": hybrid_ms + rerank_ms,
        }

        for method in methods:
            scores = evaluate_ranking(
                rankings[method],
                relevant_ids,
                k,
            )

            per_method[method].append({
                "question": query,
                "retrieved_chunk_ids": rankings[method],
                "relevant_chunk_ids": relevant_ids,
                "metrics": scores,
                "latency_ms": round(timings[method], 3),
            })

            latencies[method].append(timings[method])

    summary = {}

    for method, records in per_method.items():
        metric_names = records[0]["metrics"].keys()

        summary[method] = {
            "retrieval_metrics": {
                metric: round(
                    mean(item["metrics"][metric] for item in records), 4
                )
                for metric in metric_names
            },
            "latency": _latency_summary(latencies[method]),
            "queries_per_second": round(
                1000 / mean(latencies[method]), 3
            ) if mean(latencies[method]) > 0 else 0.0,
        }

    return {
        "evaluation": "TrustRAG Retrieval Ablation Study",
        "query_count": len(dataset),
        "k": k,
        "candidate_k": candidate_k,
        "embedding_latency": _latency_summary(
            [record["embedding_ms"] for record in []]
        ) if False else {
            "note": "Embedding time is excluded from individual retrieval method comparisons."
        },
        "summary": summary,
        "details": per_method,
    }


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        data = data.get("queries")

    if not isinstance(data, list):
        raise ValueError("Dataset must contain a list of queries.")

    return data


def save_report(
    report: dict[str, Any],
    path: str | Path = "data/evaluation/retrieval_report.json",
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)


def save_ablation_csv(
    report: dict[str, Any],
    path: str | Path = "data/evaluation/ablation_comparison.csv",
) -> None:
    import csv

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Method",
            "Precision@K",
            "Recall@K",
            "MRR",
            "NDCG@K",
            "Hit Rate@K",
            "Mean Latency (ms)",
            "Median Latency (ms)",
            "Std Latency (ms)",
            "Queries Per Second",
        ])

        for method, values in report["summary"].items():
            metrics = values["retrieval_metrics"]
            latency = values["latency"]

            writer.writerow([
                method,
                metrics.get(f"precision@{report['k']}", 0),
                metrics.get(f"recall@{report['k']}", 0),
                metrics.get("mrr", 0),
                metrics.get(f"ndcg@{report['k']}", 0),
                metrics.get(f"hit_rate@{report['k']}", 0),
                latency["mean_ms"],
                latency["median_ms"],
                latency["std_ms"],
                values["queries_per_second"],
            ])
