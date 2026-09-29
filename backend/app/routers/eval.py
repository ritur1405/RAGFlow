"""Generation-evaluation API.

Flow for a single question, per RAG configuration:

    question
      -> PIPELINE_REGISTRY[config]        (real retrieval + real generation)
      -> retrieved context + answer + citations
      -> run_generation_eval              (real Gemini LLM-as-a-judge)
      -> four metrics + weighted overall
      -> EvalResult / EvalRun rows
      -> dashboard reads /compare and /results/{run_id}

There is no simulator and no placeholder scoring anywhere in this path. If a
pipeline or the judge fails, the run is marked "failed" with the error message
recorded and the request returns an error status. The dashboard therefore shows
either measured numbers or nothing -- never numbers that look measured.

The RAG pipelines are synchronous and need a Session. They are dispatched to a
worker thread via anyio.to_thread.run_sync, each with a short-lived session of
its own, so the event loop is never blocked and no Session is shared across
threads. The request-scoped session is used only for persistence.
"""

from __future__ import annotations

import enum
import logging
import uuid
from datetime import datetime
from typing import Optional

import anyio.to_thread
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import SessionLocal, get_db
from app.eval_generation import (
    ContextChunk,
    JudgeError,
    JudgeUnavailableError,
    judge_available,
    run_generation_eval,
)
from app.models import EvalResult, EvalRun, utcnow
from app.rag_pipelines import PIPELINE_REGISTRY, PipelineResult

logger = logging.getLogger("ragops.eval_router")

router = APIRouter(prefix="/api/eval", tags=["evaluation"])


# --------------------------------------------------------------------------- #
# RAG configuration enum
# --------------------------------------------------------------------------- #


class RAGConfig(str, enum.Enum):
    BASELINE = "baseline"
    DENSE = "dense"
    HYBRID = "hybrid"
    HYBRID_RERANKER = "hybrid_reranker"


RAG_CONFIG_LABELS = {
    RAGConfig.BASELINE: "Baseline RAG",
    RAGConfig.DENSE: "Dense RAG",
    RAGConfig.HYBRID: "Hybrid RAG (Dense + BM25)",
    RAGConfig.HYBRID_RERANKER: "Hybrid + Reranker",
}

# Fail fast at import if the enum and the pipeline registry ever drift apart,
# rather than discovering it mid-run as a KeyError.
assert {c.value for c in RAGConfig} == set(PIPELINE_REGISTRY), (
    "RAGConfig values and PIPELINE_REGISTRY keys must match: "
    f"{sorted(c.value for c in RAGConfig)} vs {sorted(PIPELINE_REGISTRY)}"
)


# --------------------------------------------------------------------------- #
# Request / response schemas
# --------------------------------------------------------------------------- #


class BenchmarkQuestion(BaseModel):
    question: str = Field(..., min_length=1)
    ground_truth: Optional[str] = None


class EvalRunRequest(BaseModel):
    dataset_name: str = Field(..., description="Label for the benchmark dataset used.")
    questions: list[BenchmarkQuestion] = Field(
        ..., min_length=1, description="Benchmark question set to evaluate against."
    )
    rag_configs: list[RAGConfig] = Field(
        default_factory=lambda: list(RAGConfig),
        description="Which RAG configurations to evaluate. Defaults to all four.",
    )


class EvalRunSummary(BaseModel):
    run_id: str
    rag_config: RAGConfig
    rag_config_label: str
    dataset_name: str
    num_questions: int
    status: str
    error_message: Optional[str] = None
    avg_faithfulness: Optional[float]
    avg_answer_relevance: Optional[float]
    avg_context_relevance: Optional[float]
    avg_citation_accuracy: Optional[float]
    avg_overall_score: Optional[float]
    avg_latency_ms: Optional[float]
    created_at: datetime
    completed_at: Optional[datetime]


class EvalRunResponse(BaseModel):
    runs: list[EvalRunSummary]


class MetricDetail(BaseModel):
    score: float
    reasoning: Optional[str]


class EvalResultDetail(BaseModel):
    id: str
    question: str
    ground_truth: Optional[str]
    generated_answer: str
    context_chunk_ids: list[str]
    cited_chunk_ids: list[str]
    faithfulness: MetricDetail
    answer_relevance: MetricDetail
    context_relevance: MetricDetail
    citation_accuracy: MetricDetail
    overall_score: float
    pipeline_latency_ms: Optional[float]
    judge_latency_ms: Optional[float]


class EvalResultsResponse(BaseModel):
    run: EvalRunSummary
    results: list[EvalResultDetail]


class CompareRow(BaseModel):
    rag_config: RAGConfig
    rag_config_label: str
    run_id: Optional[str]
    dataset_name: Optional[str]
    num_questions: int
    avg_faithfulness: Optional[float]
    avg_answer_relevance: Optional[float]
    avg_context_relevance: Optional[float]
    avg_citation_accuracy: Optional[float]
    avg_overall_score: Optional[float]
    avg_latency_ms: Optional[float]
    created_at: Optional[datetime]


class CompareResponse(BaseModel):
    configs: list[CompareRow]


# --------------------------------------------------------------------------- #
# Pipeline execution
# --------------------------------------------------------------------------- #


def _run_pipeline_sync(config_value: str, question: str) -> PipelineResult:
    """Runs one RAG configuration in a worker thread with its own Session.

    A SQLAlchemy Session is not thread-safe, so the request-scoped session is
    deliberately not passed in here. This session is opened and closed entirely
    inside the worker thread.
    """
    pipeline = PIPELINE_REGISTRY[config_value]
    db = SessionLocal()
    try:
        return pipeline(db, question)
    finally:
        db.close()


async def run_pipeline(config: RAGConfig, question: str) -> PipelineResult:
    """Awaitable wrapper: executes the synchronous pipeline off the event loop."""
    return await anyio.to_thread.run_sync(_run_pipeline_sync, config.value, question)


def _summary_from_run(run: EvalRun) -> EvalRunSummary:
    config = RAGConfig(run.rag_config)
    return EvalRunSummary(
        run_id=run.id,
        rag_config=config,
        rag_config_label=RAG_CONFIG_LABELS[config],
        dataset_name=run.dataset_name,
        num_questions=run.num_questions,
        status=run.status,
        error_message=run.error_message,
        avg_faithfulness=run.avg_faithfulness,
        avg_answer_relevance=run.avg_answer_relevance,
        avg_context_relevance=run.avg_context_relevance,
        avg_citation_accuracy=run.avg_citation_accuracy,
        avg_overall_score=run.avg_overall_score,
        avg_latency_ms=run.avg_latency_ms,
        created_at=run.created_at,
        completed_at=run.completed_at,
    )


def _mark_failed(db: Session, run: EvalRun, exc: Exception) -> None:
    """Records the failure on the run so it is visible, then leaves it failed."""
    run.status = "failed"
    run.error_message = f"{type(exc).__name__}: {exc}"
    run.completed_at = utcnow()
    run.avg_faithfulness = None
    run.avg_answer_relevance = None
    run.avg_context_relevance = None
    run.avg_citation_accuracy = None
    run.avg_overall_score = None
    run.avg_latency_ms = None
    db.commit()


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@router.post("/run", response_model=EvalRunResponse, status_code=201)
async def run_evaluation(payload: EvalRunRequest, db: Session = Depends(get_db)):
    """Evaluates the benchmark questions against each requested RAG configuration."""
    configs = payload.rag_configs or list(RAGConfig)

    # Pre-flight: refuse up front rather than creating runs that cannot be
    # scored. Without this the caller would get a half-populated set of runs.
    if not judge_available():
        raise HTTPException(
            status_code=503,
            detail=(
                "Evaluation unavailable: the Gemini judge is not configured. "
                "Set GEMINI_API_KEY and retry."
            ),
        )

    completed: list[EvalRun] = []

    for config in configs:
        run = EvalRun(
            id=str(uuid.uuid4()),
            rag_config=config.value,
            dataset_name=payload.dataset_name,
            num_questions=len(payload.questions),
            status="running",
            created_at=utcnow(),
        )
        db.add(run)
        db.commit()

        totals = {
            "faithfulness": 0.0,
            "answer_relevance": 0.0,
            "context_relevance": 0.0,
            "citation_accuracy": 0.0,
            "overall": 0.0,
            "latency": 0.0,
        }

        try:
            for q in payload.questions:
                # 1. Real RAG pipeline, off the event loop, own session.
                pipeline_out = await run_pipeline(config, q.question)

                # 2. Real LLM judge over the actual retrieved context.
                context_chunks = [
                    ContextChunk(chunk_id=str(doc.id), text=doc.content)
                    for doc in pipeline_out.context_docs
                ]
                eval_result = await run_generation_eval(
                    query=q.question,
                    answer=pipeline_out.answer,
                    context_chunks=context_chunks,
                    cited_chunk_ids=pipeline_out.cited_chunk_ids,
                )

                # 3. Persist the measurement.
                db.add(
                    EvalResult(
                        id=str(uuid.uuid4()),
                        run_id=run.id,
                        question=q.question,
                        ground_truth=q.ground_truth,
                        generated_answer=pipeline_out.answer,
                        context_chunk_ids=pipeline_out.retrieved_chunk_ids,
                        cited_chunk_ids=pipeline_out.cited_chunk_ids,
                        faithfulness_score=eval_result.faithfulness.score,
                        faithfulness_reasoning=eval_result.faithfulness.reasoning,
                        answer_relevance_score=eval_result.answer_relevance.score,
                        answer_relevance_reasoning=eval_result.answer_relevance.reasoning,
                        context_relevance_score=eval_result.context_relevance.score,
                        context_relevance_reasoning=eval_result.context_relevance.reasoning,
                        citation_accuracy_score=eval_result.citation_accuracy.score,
                        citation_accuracy_reasoning=eval_result.citation_accuracy.reasoning,
                        overall_score=eval_result.overall_score,
                        pipeline_latency_ms=pipeline_out.latency_ms,
                        judge_latency_ms=eval_result.total_latency_ms,
                    )
                )

                totals["faithfulness"] += eval_result.faithfulness.score
                totals["answer_relevance"] += eval_result.answer_relevance.score
                totals["context_relevance"] += eval_result.context_relevance.score
                totals["citation_accuracy"] += eval_result.citation_accuracy.score
                totals["overall"] += eval_result.overall_score
                totals["latency"] += pipeline_out.latency_ms

            n = len(payload.questions)
            run.avg_faithfulness = totals["faithfulness"] / n
            run.avg_answer_relevance = totals["answer_relevance"] / n
            run.avg_context_relevance = totals["context_relevance"] / n
            run.avg_citation_accuracy = totals["citation_accuracy"] / n
            run.avg_overall_score = totals["overall"] / n
            run.avg_latency_ms = totals["latency"] / n
            run.status = "completed"
            run.completed_at = utcnow()
            db.commit()
            completed.append(run)

        except JudgeUnavailableError as exc:
            logger.exception("Judge unavailable during run %s (%s)", run.id, config.value)
            _mark_failed(db, run, exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        except JudgeError as exc:
            logger.exception("Judge failed during run %s (%s)", run.id, config.value)
            _mark_failed(db, run, exc)
            raise HTTPException(
                status_code=502,
                detail=f"Evaluation failed for config '{config.value}': {exc}",
            ) from exc

        except Exception as exc:
            # Pipeline failure (retrieval, generation, reranker, database).
            # Logged with traceback, recorded on the run, surfaced as an error.
            logger.exception("Pipeline failed during run %s (%s)", run.id, config.value)
            _mark_failed(db, run, exc)
            raise HTTPException(
                status_code=502,
                detail=f"RAG pipeline failed for config '{config.value}': {exc}",
            ) from exc

    return EvalRunResponse(runs=[_summary_from_run(r) for r in completed])


@router.get("/compare", response_model=CompareResponse)
@router.get("", response_model=CompareResponse)
@router.get("/", response_model=CompareResponse)
def compare_configs(
    run_ids: Optional[str] = Query(None, description="Optional comma-separated run IDs"),
    db: Session = Depends(get_db),
):
    """Side-by-side aggregate comparison, one row per configuration.

    Only completed runs are considered, so a failed run never contributes
    numbers to the comparison. A configuration with no completed run reports
    nulls rather than zeros, letting the dashboard distinguish "not measured"
    from "measured as zero".
    """
    explicit_ids = [r.strip() for r in run_ids.split(",")] if run_ids else None
    rows: list[CompareRow] = []

    for config in RAGConfig:
        stmt = select(EvalRun).where(
            EvalRun.rag_config == config.value, EvalRun.status == "completed"
        )
        if explicit_ids:
            stmt = stmt.where(EvalRun.id.in_(explicit_ids))
        run = db.execute(stmt.order_by(EvalRun.created_at.desc())).scalars().first()

        if run is None:
            rows.append(
                CompareRow(
                    rag_config=config,
                    rag_config_label=RAG_CONFIG_LABELS[config],
                    run_id=None,
                    dataset_name=None,
                    num_questions=0,
                    avg_faithfulness=None,
                    avg_answer_relevance=None,
                    avg_context_relevance=None,
                    avg_citation_accuracy=None,
                    avg_overall_score=None,
                    avg_latency_ms=None,
                    created_at=None,
                )
            )
        else:
            rows.append(
                CompareRow(
                    rag_config=config,
                    rag_config_label=RAG_CONFIG_LABELS[config],
                    run_id=run.id,
                    dataset_name=run.dataset_name,
                    num_questions=run.num_questions,
                    avg_faithfulness=run.avg_faithfulness,
                    avg_answer_relevance=run.avg_answer_relevance,
                    avg_context_relevance=run.avg_context_relevance,
                    avg_citation_accuracy=run.avg_citation_accuracy,
                    avg_overall_score=run.avg_overall_score,
                    avg_latency_ms=run.avg_latency_ms,
                    created_at=run.created_at,
                )
            )

    return CompareResponse(configs=rows)


@router.get("/results/{run_id}", response_model=EvalResultsResponse)
def get_results(run_id: str, db: Session = Depends(get_db)):
    """Per-question detail for one run, including each metric's reasoning."""
    run = db.get(EvalRun, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Eval run '{run_id}' not found.")

    rows = (
        db.execute(select(EvalResult).where(EvalResult.run_id == run_id))
        .scalars()
        .all()
    )

    details = [
        EvalResultDetail(
            id=row.id,
            question=row.question,
            ground_truth=row.ground_truth,
            generated_answer=row.generated_answer,
            context_chunk_ids=row.context_chunk_ids or [],
            cited_chunk_ids=row.cited_chunk_ids or [],
            faithfulness=MetricDetail(
                score=row.faithfulness_score, reasoning=row.faithfulness_reasoning
            ),
            answer_relevance=MetricDetail(
                score=row.answer_relevance_score,
                reasoning=row.answer_relevance_reasoning,
            ),
            context_relevance=MetricDetail(
                score=row.context_relevance_score,
                reasoning=row.context_relevance_reasoning,
            ),
            citation_accuracy=MetricDetail(
                score=row.citation_accuracy_score,
                reasoning=row.citation_accuracy_reasoning,
            ),
            overall_score=row.overall_score,
            pipeline_latency_ms=row.pipeline_latency_ms,
            judge_latency_ms=row.judge_latency_ms,
        )
        for row in rows
    ]

    return EvalResultsResponse(run=_summary_from_run(run), results=details)
