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
from sqlalchemy.orm import Session

from app.database import SessionLocal, get_db
from app.models import ExperimentConfig, ExperimentRun, utcnow
from app.rag_pipelines import RETRIEVER_REGISTRY, RETRIEVER_SCORE_LABELS, Ranked

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
    Strategy.HYBRID_RERANKER: "Hybrid + Reranker",
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
