"""Retrieval-only benchmark: every retriever scored on the shared dataset.

The plan's Week 4 deliverable is a Dense vs BM25 vs Hybrid comparison, and the
Week 5 retrieval metrics should be measurable on their own. The end-to-end
evaluation (/api/eval) also generates and judges an answer for every
question, which costs several LLM calls each; this benchmark stops after
retrieval, so it costs no generation at all.

Cost control: each benchmark question is embedded once per process and the
vector is reused for every strategy and every later run (see
query_embeddings). A full run of all five strategies therefore makes at most
one embedding call per question, and repeat runs make none.

Latency is measured around the retriever call only. With precomputed query
embeddings that excludes the embedding request, so the figure compares the
retrieval algorithms themselves; the embedding latency is reported
separately, as it is the same for every dense-based strategy.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.config import EMBED_TASK_QUERY, EMBEDDING_MODEL
from app.database import SessionLocal
from app.eval_retrieval import RetrievalScores, score_retrieval
from app.rag_pipelines import RETRIEVER_REGISTRY, precomputed_query_embeddings
from app.services import generate_embedding

logger = logging.getLogger("ragops.retrieval_benchmark")

METRICS = ("precision_at_k", "recall_at_k", "reciprocal_rank", "ndcg_at_k")

# (embedding model, question) -> vector. Process-wide, so repeat runs and
# parameter sweeps never re-embed a question.
_embedding_cache: Dict[tuple, List[float]] = {}
_embedding_lock = threading.Lock()


@dataclass
class EmbeddingStats:
    calls_made: int = 0
    reused: int = 0
    latencies_ms: List[float] = field(default_factory=list)

    @property
    def avg_latency_ms(self) -> Optional[float]:
        return sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else None


def query_embeddings(questions: List[str]) -> tuple[Dict[str, List[float]], EmbeddingStats]:
    """Vectors for every question, embedding only those not seen before."""
    stats = EmbeddingStats()
    vectors: Dict[str, List[float]] = {}
    for question in questions:
        key = (EMBEDDING_MODEL, question)
        with _embedding_lock:
            cached = _embedding_cache.get(key)
        if cached is not None:
            stats.reused += 1
        else:
            start = time.perf_counter()
            cached = generate_embedding(question, task_type=EMBED_TASK_QUERY)
            stats.latencies_ms.append((time.perf_counter() - start) * 1000)
            stats.calls_made += 1
            with _embedding_lock:
                _embedding_cache[key] = cached
        vectors[question] = cached
    return vectors, stats


def clear_embedding_cache() -> None:
    with _embedding_lock:
        _embedding_cache.clear()


@dataclass
class QuestionResult:
    question: str
    type: Optional[str]
    relevant_pages: List[int]
    retrieved_pages: List[Optional[int]]
    latency_ms: float
    scores: RetrievalScores

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "type": self.type,
            "relevant_pages": self.relevant_pages,
            "retrieved_pages": self.retrieved_pages,
            "latency_ms": round(self.latency_ms, 2),
            **{m: round(getattr(self.scores, m), 4) for m in METRICS},
        }


def run_strategy(
    strategy: str, questions: List[dict], top_k: int, embeddings: Dict[str, List[float]]
) -> List[QuestionResult]:
    """Scores one retriever on every question. Blocking; call from a worker thread.

    Raises on the first retrieval error: a strategy that could not run is
    reported as failed, never as a partial average.
    """
    retriever = RETRIEVER_REGISTRY[strategy]
    db = SessionLocal()
    results: List[QuestionResult] = []
    try:
        with precomputed_query_embeddings(embeddings):
            for q in questions:
                start = time.perf_counter()
                ranked = retriever(db, q["question"], top_k=top_k)
                latency_ms = (time.perf_counter() - start) * 1000
                docs = [doc for doc, _score in ranked]
                results.append(
                    QuestionResult(
                        question=q["question"],
                        type=q.get("type"),
                        relevant_pages=list(q["relevant_pages"]),
                        retrieved_pages=[doc.page_number for doc in docs[:top_k]],
                        latency_ms=latency_ms,
                        scores=score_retrieval(docs, q.get("source_file"), q["relevant_pages"], k=top_k),
                    )
                )
    finally:
        db.close()
    return results


def aggregate(results: List[QuestionResult]) -> dict:
    """Mean of each metric and of latency, overall and per question type."""

    def means(rows: List[QuestionResult]) -> dict:
        n = len(rows)
        return {"n": n, **{m: sum(getattr(r.scores, m) for r in rows) / n for m in METRICS}}

    by_type: Dict[str, List[QuestionResult]] = {}
    for r in results:
        by_type.setdefault(r.type or "untyped", []).append(r)

    return {
        "overall": means(results),
        "avg_latency_ms": sum(r.latency_ms for r in results) / len(results),
        "by_type": {t: means(rows) for t, rows in sorted(by_type.items())},
    }
