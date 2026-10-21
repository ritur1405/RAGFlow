"""Retrieval-experiment API (Week 4: Dense vs BM25 vs Hybrid).

Flow for one comparison:

    query + strategies + top_k
      -> RETRIEVER_REGISTRY[strategy]   (real retrieval, no generation)
      -> ranked chunks + retrieval latency, per strategy
      -> ExperimentRun rows
      -> dashboard renders the columns side by side

Retrieval only: no answer is generated and nothing is judged, so a comparison
is cheap enough to run interactively. The end-to-end, LLM-judged benchmark of
the four configurations lives in app/routers/eval.py.

Strategies run one after another rather than concurrently, so each latency
figure measures that strategy alone instead of contention between them. Each
runs in a worker thread with its own Session, as in the evaluation router.

A strategy that fails is recorded and returned as failed with its error. It is
never dropped or replaced, so the comparison cannot show a strategy that did
not actually run.

POST /benchmark does the same over the whole shared, page-labelled dataset
and scores every strategy with Recall@K, Precision@K, MRR and nDCG@K, still
without generation (see app/retrieval_benchmark.py).
"""

from __future__ import annotations

import enum
import logging
import time
import uuid
from datetime import datetime
from typing import Optional

import anyio.to_thread
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from google.genai import errors as genai_errors
from sqlalchemy.orm import Session

from app.benchmark import load_benchmark
from app.database import SessionLocal, get_db
from app.models import Document, ExperimentConfig, ExperimentRun, RetrievalBenchmarkRun, utcnow
from app.rag_pipelines import FINAL_TOP_K, RETRIEVER_REGISTRY, RETRIEVER_SCORE_LABELS, Ranked
from app.retrieval_benchmark import aggregate, query_embeddings, run_strategy

logger = logging.getLogger("ragops.experiments_router")

router = APIRouter(prefix="/api/experiments", tags=["experiments"])


# --------------------------------------------------------------------------- #
# Strategy enum
# --------------------------------------------------------------------------- #


class Strategy(str, enum.Enum):
    BASELINE = "baseline"
    DENSE = "dense"
    BM25 = "bm25"
    HYBRID = "hybrid"
    HYBRID_RERANKER = "hybrid_reranker"


STRATEGY_LABELS = {
    Strategy.BASELINE: "Baseline (dense, no gate)",
    Strategy.DENSE: "Dense",
    Strategy.BM25: "BM25",
    Strategy.HYBRID: "Hybrid (RRF)",
    Strategy.HYBRID_RERANKER: "Hybrid + Cross-Encoder",
}

# The plan's Week 4 deliverable: Dense vs BM25 vs Hybrid.
DEFAULT_STRATEGIES = [Strategy.DENSE, Strategy.BM25, Strategy.HYBRID]

assert {s.value for s in Strategy} == set(RETRIEVER_REGISTRY), (
    "Strategy values and RETRIEVER_REGISTRY keys must match: "
    f"{sorted(s.value for s in Strategy)} vs {sorted(RETRIEVER_REGISTRY)}"
)


# --------------------------------------------------------------------------- #
# Request / response schemas
# --------------------------------------------------------------------------- #


def _dedupe(strategies: list[Strategy]) -> list[Strategy]:
    return list(dict.fromkeys(strategies))


class CompareRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    strategies: Optional[list[Strategy]] = Field(
        None,
        min_length=1,
        description="Strategies to compare. Defaults to dense, bm25 and hybrid, "
        "or to the saved config's strategies when config_id is given.",
    )
    top_k: Optional[int] = Field(
        None, ge=1, le=20, description="Defaults to 5, or to the saved config's top_k."
    )
    config_id: Optional[str] = Field(
        None, description="Saved ExperimentConfig to run and link the runs to."
    )

    @field_validator("query")
    @classmethod
    def _strip_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("query must not be blank")
        return v

    @field_validator("strategies")
    @classmethod
    def _unique(cls, v):
        return _dedupe(v) if v else v


class RetrievedChunk(BaseModel):
    rank: int
    chunk_id: int
    file_name: Optional[str]
    title: Optional[str]
    page_number: Optional[int]
    chunk_index: Optional[int]
    score: float
    content: str


class StrategyResult(BaseModel):
    strategy: Strategy
    label: str
    score_label: str
    run_id: str
    status: str
    error_message: Optional[str] = None
    latency_ms: Optional[float]
    chunks: list[RetrievedChunk]


class CompareResponse(BaseModel):
    query: str
    top_k: int
    config_id: Optional[str]
    results: list[StrategyResult]


class ConfigCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    strategies: list[Strategy] = Field(default_factory=lambda: list(DEFAULT_STRATEGIES), min_length=1)
    top_k: int = Field(5, ge=1, le=20)

    @field_validator("name")
    @classmethod
    def _strip_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("name must not be blank")
        return v

    @field_validator("strategies")
    @classmethod
    def _unique(cls, v):
        return _dedupe(v)


class ConfigOut(BaseModel):
    id: str
    name: str
    strategies: list[Strategy]
    top_k: int
    created_at: datetime


class RunOut(BaseModel):
    run_id: str
    config_id: Optional[str]
    query: str
    strategy: Strategy
    label: str
    top_k: int
    status: str
    error_message: Optional[str]
    latency_ms: Optional[float]
    num_results: int
    top_chunk_ids: list[int]
    created_at: datetime


# --------------------------------------------------------------------------- #
# Retrieval execution
# --------------------------------------------------------------------------- #


def _retrieve_sync(strategy_value: str, query: str, top_k: int) -> tuple[Ranked, float]:
    """Runs one retriever in a worker thread with its own Session and times it."""
    retriever = RETRIEVER_REGISTRY[strategy_value]
    db = SessionLocal()
    try:
        start = time.perf_counter()
        ranked = retriever(db, query, top_k=top_k)
        latency_ms = (time.perf_counter() - start) * 1000
        return ranked, round(latency_ms, 2)
    finally:
        db.close()


async def run_retriever(strategy: Strategy, query: str, top_k: int) -> tuple[Ranked, float]:
    """Awaitable wrapper: executes the synchronous retriever off the event loop."""
    return await anyio.to_thread.run_sync(_retrieve_sync, strategy.value, query, top_k)


def _config_out(config: ExperimentConfig) -> ConfigOut:
    return ConfigOut(
        id=config.id,
        name=config.name,
        strategies=[Strategy(s) for s in config.strategies],
        top_k=config.top_k,
        created_at=config.created_at,
    )


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


@router.post("/compare", response_model=CompareResponse)
async def compare_strategies(payload: CompareRequest, db: Session = Depends(get_db)):
    """Runs each strategy's retrieval for the query and returns the rankings side by side."""
    config: Optional[ExperimentConfig] = None
    if payload.config_id:
        config = db.get(ExperimentConfig, payload.config_id)
        if config is None:
            raise HTTPException(
                status_code=404,
                detail=f"Experiment config '{payload.config_id}' not found.",
            )

    strategies = payload.strategies or (
        [Strategy(s) for s in config.strategies] if config else list(DEFAULT_STRATEGIES)
    )
    top_k = payload.top_k or (config.top_k if config else 5)

    results: list[StrategyResult] = []
    for strategy in strategies:
        run = ExperimentRun(
            id=str(uuid.uuid4()),
            config_id=config.id if config else None,
            query=payload.query,
            strategy=strategy.value,
            top_k=top_k,
            created_at=utcnow(),
        )
        chunks: list[RetrievedChunk] = []
        try:
            ranked, latency_ms = await run_retriever(strategy, payload.query, top_k)
            chunks = [
                RetrievedChunk(
                    rank=rank,
                    chunk_id=doc.id,
                    file_name=doc.file_name,
                    title=doc.title,
                    page_number=doc.page_number,
                    chunk_index=doc.chunk_index,
                    score=round(float(score), 6),
                    content=doc.content,
                )
                for rank, (doc, score) in enumerate(ranked, start=1)
            ]
            run.status = "completed"
            run.latency_ms = latency_ms
            run.results = [
                {
                    "chunk_id": c.chunk_id,
                    "file_name": c.file_name,
                    "page_number": c.page_number,
                    "score": c.score,
                }
                for c in chunks
            ]
        except Exception as exc:
            logger.exception("Retriever %s failed for query=%r", strategy.value, payload.query)
            run.status = "failed"
            run.error_message = f"{type(exc).__name__}: {exc}"

        db.add(run)
        db.commit()

        results.append(
            StrategyResult(
                strategy=strategy,
                label=STRATEGY_LABELS[strategy],
                score_label=RETRIEVER_SCORE_LABELS[strategy.value],
                run_id=run.id,
                status=run.status,
                error_message=run.error_message,
                latency_ms=run.latency_ms,
                chunks=chunks,
            )
        )

    return CompareResponse(
        query=payload.query,
        top_k=top_k,
        config_id=config.id if config else None,
        results=results,
    )


@router.get("/runs", response_model=list[RunOut])
def list_runs(
    limit: int = Query(50, ge=1, le=500),
    config_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """Most recent experiment runs first."""
    stmt = select(ExperimentRun).order_by(ExperimentRun.created_at.desc()).limit(limit)
    if config_id:
        stmt = stmt.where(ExperimentRun.config_id == config_id)
    runs = db.execute(stmt).scalars().all()
    return [
        RunOut(
            run_id=run.id,
            config_id=run.config_id,
            query=run.query,
            strategy=Strategy(run.strategy),
            label=STRATEGY_LABELS[Strategy(run.strategy)],
            top_k=run.top_k,
            status=run.status,
            error_message=run.error_message,
            latency_ms=run.latency_ms,
            num_results=len(run.results or []),
            top_chunk_ids=[r["chunk_id"] for r in (run.results or [])],
            created_at=run.created_at,
        )
        for run in runs
    ]


@router.post("/configs", response_model=ConfigOut, status_code=201)
def create_config(payload: ConfigCreate, db: Session = Depends(get_db)):
    """Saves a named, reusable experiment configuration."""
    config = ExperimentConfig(
        id=str(uuid.uuid4()),
        name=payload.name,
        strategies=[s.value for s in payload.strategies],
        top_k=payload.top_k,
        created_at=utcnow(),
    )
    db.add(config)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=409, detail=f"An experiment config named '{payload.name}' already exists."
        )
    return _config_out(config)


@router.get("/configs", response_model=list[ConfigOut])
def list_configs(db: Session = Depends(get_db)):
    configs = (
        db.execute(select(ExperimentConfig).order_by(ExperimentConfig.created_at.desc()))
        .scalars()
        .all()
    )
    return [_config_out(c) for c in configs]


@router.delete("/configs/{config_id}", status_code=204)
def delete_config(config_id: str, db: Session = Depends(get_db)):
    """Deletes a saved config. Its past runs are kept and unlinked."""
    config = db.get(ExperimentConfig, config_id)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Experiment config '{config_id}' not found.")
    for run in config.runs:
        run.config_id = None
    db.delete(config)
    db.commit()


# --------------------------------------------------------------------------- #
# Retrieval benchmark: every strategy scored on the shared dataset
# --------------------------------------------------------------------------- #


class BenchmarkRequest(BaseModel):
    strategies: list[Strategy] = Field(
        default_factory=lambda: list(Strategy),
        min_length=1,
        description="Strategies to score. Defaults to all five.",
    )
    top_k: int = Field(FINAL_TOP_K, ge=1, le=20, description="K for every metric.")

    @field_validator("strategies")
    @classmethod
    def _unique(cls, v):
        return _dedupe(v)


class MetricMeans(BaseModel):
    n: int
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float


class StrategyBenchmark(BaseModel):
    run_id: str
    strategy: Strategy
    label: str
    status: str
    error_message: Optional[str] = None
    k: int
    num_questions: int
    avg_precision_at_k: Optional[float] = None
    avg_recall_at_k: Optional[float] = None
    avg_mrr: Optional[float] = None
    avg_ndcg_at_k: Optional[float] = None
    avg_latency_ms: Optional[float] = None
    by_type: dict[str, MetricMeans] = Field(default_factory=dict)
    created_at: datetime


class EmbeddingUsage(BaseModel):
    calls_made: int
    reused: int
    avg_latency_ms: Optional[float]


class BenchmarkResponse(BaseModel):
    batch_id: str
    dataset_name: str
    k: int
    num_questions: int
    embedding: EmbeddingUsage
    results: list[StrategyBenchmark]
    # One row per question per strategy, for drill-down.
    per_question: dict[str, list[dict]] = Field(default_factory=dict)


def _benchmark_out(run: RetrievalBenchmarkRun) -> StrategyBenchmark:
    strategy = Strategy(run.strategy)
    return StrategyBenchmark(
        run_id=run.id,
        strategy=strategy,
        label=STRATEGY_LABELS[strategy],
        status=run.status,
        error_message=run.error_message,
        k=run.k,
        num_questions=run.num_questions,
        avg_precision_at_k=run.avg_precision_at_k,
        avg_recall_at_k=run.avg_recall_at_k,
        avg_mrr=run.avg_mrr,
        avg_ndcg_at_k=run.avg_ndcg_at_k,
        avg_latency_ms=run.avg_latency_ms,
        by_type=run.by_type or {},
        created_at=run.created_at,
    )


@router.post("/benchmark", response_model=BenchmarkResponse)
async def run_retrieval_benchmark(payload: BenchmarkRequest, db: Session = Depends(get_db)):
    """Scores each retriever on the shared, page-labelled benchmark.

    Retrieval only: no answer is generated, so this makes no LLM calls, and
    each question is embedded at most once per server process.
    """
    try:
        dataset = load_benchmark()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Benchmark dataset not found.")

    questions = [q for q in dataset["questions"] if q.get("relevant_pages")]
    if not questions:
        raise HTTPException(status_code=422, detail="The benchmark has no labelled questions.")

    # Without the labelled documents indexed every score would be zero, which
    # would look like a measurement rather than a missing upload.
    source_files = {q["source_file"] for q in questions if q.get("source_file")}
    indexed = set(
        db.execute(
            select(Document.file_name).where(Document.file_name.in_(source_files)).distinct()
        ).scalars()
    )
    missing = sorted(source_files - indexed)
    if missing:
        raise HTTPException(
            status_code=409,
            detail=f"Upload the benchmark document(s) first: {', '.join(missing)}.",
        )

    try:
        embeddings, embed_stats = await anyio.to_thread.run_sync(
            query_embeddings, [q["question"] for q in questions]
        )
    except genai_errors.APIError as exc:
        status = 503 if exc.code == 429 else 502
        raise HTTPException(
            status_code=status, detail=f"Embedding the benchmark questions failed: {exc.message}"
        )

    batch_id = str(uuid.uuid4())
    results: list[StrategyBenchmark] = []
    per_question: dict[str, list[dict]] = {}
    for strategy in payload.strategies:
        run = RetrievalBenchmarkRun(
            id=str(uuid.uuid4()),
            batch_id=batch_id,
            dataset_name=dataset["dataset_name"],
            strategy=strategy.value,
            k=payload.top_k,
            num_questions=len(questions),
            created_at=utcnow(),
        )
        try:
            rows = await anyio.to_thread.run_sync(
                run_strategy, strategy.value, questions, payload.top_k, embeddings
            )
            summary = aggregate(rows)
            overall = summary["overall"]
            run.status = "completed"
            run.avg_precision_at_k = overall["precision_at_k"]
            run.avg_recall_at_k = overall["recall_at_k"]
            run.avg_mrr = overall["reciprocal_rank"]
            run.avg_ndcg_at_k = overall["ndcg_at_k"]
            run.avg_latency_ms = summary["avg_latency_ms"]
            run.by_type = summary["by_type"]
            run.per_question = [r.to_dict() for r in rows]
            per_question[strategy.value] = run.per_question
        except Exception as exc:
            logger.exception("Retrieval benchmark failed for %s", strategy.value)
            run.status = "failed"
            run.error_message = f"{type(exc).__name__}: {exc}"
        db.add(run)
        db.commit()
        results.append(_benchmark_out(run))

    return BenchmarkResponse(
        batch_id=batch_id,
        dataset_name=dataset["dataset_name"],
        k=payload.top_k,
        num_questions=len(questions),
        embedding=EmbeddingUsage(
            calls_made=embed_stats.calls_made,
            reused=embed_stats.reused,
            avg_latency_ms=embed_stats.avg_latency_ms,
        ),
        results=results,
        per_question=per_question,
    )


@router.get("/benchmark/runs", response_model=list[StrategyBenchmark])
def list_benchmark_runs(
    limit: int = Query(50, ge=1, le=500), db: Session = Depends(get_db)
):
    """Most recent retrieval-benchmark results first."""
    runs = (
        db.execute(
            select(RetrievalBenchmarkRun)
            .order_by(RetrievalBenchmarkRun.created_at.desc())
            .limit(limit)
        )
        .scalars()
        .all()
    )
    return [_benchmark_out(run) for run in runs]
