from __future__ import annotations

import hashlib
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.pipeline import TrustRAGPipeline
from src.verification.verification_pipeline import ClaimVerificationPipeline


BASE_DIR = Path(__file__).resolve().parents[2]
UPLOAD_DIR = BASE_DIR / "data" / "raw" / "uploads"
LIBRARY_DIR = BASE_DIR / "data" / "processed" / "library"

MAX_UPLOAD_BYTES = 50 * 1024 * 1024

app = FastAPI(
    title="TrustRAG API",
    description="Evidence-grounded scientific research assistant.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline = None
_pipeline_lock = Lock()


class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    candidate_k: int = 30


def get_pipeline() -> TrustRAGPipeline:
    """Initialize the models only when the first operation needs them."""
    global _pipeline

    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                pipeline = TrustRAGPipeline(
                    verification_pipeline=ClaimVerificationPipeline()
                )

                if (LIBRARY_DIR / "library.json").exists():
                    pipeline.load_library(LIBRARY_DIR)

                _pipeline = pipeline

    return _pipeline


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "TrustRAG API",
        "version": "0.1.0",
        "pipeline_loaded": _pipeline is not None,
    }


@app.get("/documents")
def list_documents():
    if _pipeline is None:
        if not (LIBRARY_DIR / "library.json").exists():
            return {"documents": [], "total": 0}

    pipeline = get_pipeline()

    documents = [
        {
            "document_id": result["document_id"],
            "file_name": result["file_name"],
            "page_count": result["page_count"],
            "section_count": result["section_count"],
            "chunk_count": result["chunk_count"],
        }
        for result in pipeline.ingested_files.values()
    ]

    return {"documents": documents, "total": len(documents)}


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or Path(file.filename).suffix.lower() != ".pdf":
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted.",
        )

    content = await file.read(MAX_UPLOAD_BYTES + 1)

    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="PDF exceeds the 50 MB upload limit.",
        )

    if b"%PDF-" not in content[:1024]:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file does not appear to be a valid PDF.",
        )

    file_hash = hashlib.sha256(content).hexdigest()
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    safe_path = UPLOAD_DIR / f"{file_hash}.pdf"

    if not safe_path.exists():
        safe_path.write_bytes(content)

    try:
        pipeline = get_pipeline()
        result = pipeline.ingest_pdf(safe_path)
        pipeline.save_library(LIBRARY_DIR)

        return {
            "message": "PDF indexed successfully.",
            **result,
        }

    except (ValueError, FileNotFoundError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Document ingestion failed: {error}",
        ) from error


@app.post("/query")
def query_documents(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    if request.top_k <= 0 or request.candidate_k <= 0:
        raise HTTPException(
            status_code=400,
            detail="Retrieval limits must be positive.",
        )

    try:
        pipeline = get_pipeline()
        return pipeline.ask(
            query=request.question,
            top_k=request.top_k,
            candidate_k=request.candidate_k,
        )

    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/evaluation/report")
def get_evaluation_report():
    """Return the latest TrustRAG retrieval evaluation report."""
    import json
    from pathlib import Path
    from fastapi.responses import JSONResponse

    report_path = Path("data/evaluation/retrieval_report.json")

    if not report_path.exists():
        return JSONResponse(
            status_code=404,
            content={
                "detail": "No evaluation report available. Run the benchmark first."
            },
        )

    try:
        with report_path.open("r", encoding="utf-8") as file:
            report = json.load(file)
    except (json.JSONDecodeError, OSError):
        return JSONResponse(
            status_code=500,
            content={"detail": "Evaluation report could not be read."},
        )

    if not isinstance(report, dict) or "summary" not in report:
        return JSONResponse(
            status_code=500,
            content={"detail": "Evaluation report has an invalid format."},
        )

    return report


@app.post("/evaluation/run")
def run_evaluation_endpoint():
    """Validate the benchmark and execute retrieval experiments."""
    from fastapi import HTTPException
    from src.evaluation.runner import (
        evaluate_retrieval,
        load_dataset,
        save_report,
        save_ablation_csv,
    )
    from src.evaluation.validate_benchmark import validate_benchmark

    benchmark_path = "data/evaluation/benchmark.json"
    candidates_path = "data/evaluation/annotation_candidates.json"

    try:
        validation = validate_benchmark(
            benchmark_path,
            candidates_path,
        )

        if not validation["valid"]:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Benchmark validation failed.",
                    "validation": validation,
                },
            )

        pipeline = get_pipeline()

        if not pipeline.documents:
            raise HTTPException(
                status_code=409,
                detail="No indexed documents are loaded.",
            )

        dataset = load_dataset(benchmark_path)

        report = evaluate_retrieval(
            pipeline=pipeline,
            dataset=dataset,
            k=5,
            candidate_k=30,
        )

        save_report(report)
        save_ablation_csv(report)

        return {
            "message": "Evaluation completed.",
            "query_count": report["query_count"],
            "summary": report["summary"],
        }

    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {error}",
        )


@app.get("/evaluation/export/csv")
def export_evaluation_csv():
    from pathlib import Path
    from fastapi import HTTPException
    from fastapi.responses import FileResponse

    report = Path("data/evaluation/ablation_comparison.csv")

    if not report.is_file():
        raise HTTPException(
            status_code=404,
            detail="No ablation CSV available. Run an evaluation first."
        )

    return FileResponse(
        path=str(report),
        media_type="text/csv",
        filename="trustrag_ablation_comparison.csv",
    )


@app.get("/evaluation/export/json")
def export_evaluation_json():
    from pathlib import Path
    from fastapi import HTTPException
    from fastapi.responses import FileResponse

    report = Path("data/evaluation/retrieval_report.json")

    if not report.is_file():
        raise HTTPException(
            status_code=404,
            detail="No evaluation JSON available. Run an evaluation first."
        )

    return FileResponse(
        path=str(report),
        media_type="application/json",
        filename="trustrag_retrieval_report.json",
    )


@app.get("/evaluation/candidates")
def get_evaluation_candidates():
    import json
    from pathlib import Path
    from fastapi import HTTPException

    path = Path("data/evaluation/annotation_candidates.json")
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Candidates not found. Run prepare_benchmark first."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/evaluation/benchmark")
def get_evaluation_benchmark():
    import json
    from pathlib import Path

    path = Path("data/evaluation/benchmark.json")
    if not path.is_file():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

@app.post("/evaluation/benchmark")
def save_evaluation_benchmark(payload: list[dict]):
    import json
    from pathlib import Path
    from fastapi import HTTPException

    evaluation_dir = Path("data/evaluation")
    candidates_path = evaluation_dir / "annotation_candidates.json"
    benchmark_path = evaluation_dir / "benchmark.json"

    if not candidates_path.is_file():
        raise HTTPException(
            status_code=400,
            detail="Generate annotation candidates before saving the benchmark."
        )

    try:
        candidate_data = json.loads(candidates_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Cannot read candidates: {exc}")

    valid_ids = set()

    def collect_ids(value):
        if isinstance(value, list):
            for item in value:
                collect_ids(item)
        elif isinstance(value, dict):
            chunk_id = value.get("chunk_id")
            if chunk_id is not None:
                valid_ids.add(str(chunk_id))
            for child in value.values():
                if isinstance(child, (dict, list)):
                    collect_ids(child)

    collect_ids(candidate_data)

    if not valid_ids:
        raise HTTPException(
            status_code=400,
            detail="No chunk IDs found in annotation candidates. Regenerate the candidates."
        )

    cleaned = []
    seen_questions = set()

    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail=f"Entry {index} must be an object.")

        question = str(item.get("question", "")).strip()
        ids = item.get("relevant_chunk_ids", [])

        if not question:
            raise HTTPException(status_code=400, detail=f"Entry {index} has an empty question.")

        normalized_question = question.casefold()
        if normalized_question in seen_questions:
            raise HTTPException(
                status_code=400,
                detail=f"Duplicate question detected: {question}"
            )
        seen_questions.add(normalized_question)

        if not isinstance(ids, list) or not ids:
            raise HTTPException(
                status_code=400,
                detail=f"Question {index} must have at least one relevant chunk."
            )

        selected_ids = list(dict.fromkeys(str(value) for value in ids))
        unknown_ids = [value for value in selected_ids if value not in valid_ids]

        if unknown_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Question {index} contains unknown chunk IDs: {unknown_ids}"
            )

        cleaned.append({
            "question": question,
            "relevant_chunk_ids": selected_ids
        })

    if not cleaned:
        raise HTTPException(status_code=400, detail="Benchmark cannot be empty.")

    evaluation_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path.write_text(
        json.dumps(cleaned, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    return {
        "saved": len(cleaned),
        "validated_chunk_ids": len(valid_ids),
        "path": str(benchmark_path)
    }


@app.get("/evaluation/answer-quality")
def get_answer_quality_report():
    import json
    from pathlib import Path
    from fastapi import HTTPException

    path = Path("data/evaluation/answer_quality_results.json")
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Answer-quality report not generated yet."
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
