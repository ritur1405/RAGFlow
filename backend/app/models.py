from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from app.database import Base


def utcnow() -> datetime:
    """Timezone-aware UTC now. Shared so every timestamp is written the same way."""
    return datetime.now(timezone.utc)


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, nullable=True)     # Stores original filename (e.g. sample.pdf)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=True)  # Page from PDF
    chunk_index = Column(Integer, nullable=True)  # Position index of chunk
    embedding = Column(Vector(768))                # pgvector embedding


# ---------------------------------------------------------------------------
# Generation-evaluation tables
# ---------------------------------------------------------------------------
# One EvalRun per (dataset, RAG configuration) pair, holding the aggregate
# metrics the comparison dashboard reads. One EvalResult per question within
# that run, holding the per-metric score and the judge's reasoning so a result
# can be audited rather than taken on trust.
#
# These are the single canonical definitions. The evaluation router imports
# them from here; it does not declare its own.


class EvalRun(Base):
    __tablename__ = "eval_runs"

    id = Column(String, primary_key=True)
    rag_config = Column(String, nullable=False, index=True)
    dataset_name = Column(String, nullable=False)
    num_questions = Column(Integer, nullable=False, default=0)

    # running | completed | failed
    status = Column(String, nullable=False, default="running", index=True)
    # Populated only when status == "failed", so a failure is inspectable
    # instead of being indistinguishable from a successful run.
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), default=utcnow, index=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)

    avg_faithfulness = Column(Float, nullable=True)
    avg_answer_relevance = Column(Float, nullable=True)
    avg_context_relevance = Column(Float, nullable=True)
    avg_citation_accuracy = Column(Float, nullable=True)
    avg_overall_score = Column(Float, nullable=True)
    avg_latency_ms = Column(Float, nullable=True)

    # Retrieval metrics over the questions that carry relevance labels
    # (app/eval_retrieval.py). Null when no question in the run is labelled.
    retrieval_k = Column(Integer, nullable=True)
    num_labelled = Column(Integer, nullable=True)
    avg_precision_at_k = Column(Float, nullable=True)
    avg_recall_at_k = Column(Float, nullable=True)
    avg_mrr = Column(Float, nullable=True)
    avg_ndcg_at_k = Column(Float, nullable=True)

    results = relationship(
        "EvalResult", back_populates="run", cascade="all, delete-orphan"
    )


class EvalResult(Base):
    __tablename__ = "eval_results"

    id = Column(String, primary_key=True)
    run_id = Column(String, ForeignKey("eval_runs.id"), nullable=False, index=True)

    question = Column(Text, nullable=False)
    ground_truth = Column(Text, nullable=True)
    generated_answer = Column(Text, nullable=False)

    # Chunk ids are stored as strings so they line up with the citation
    # markers the model emits and with ContextChunk.chunk_id in the judge.
    context_chunk_ids = Column(JSON, nullable=True)   # list[str]
    cited_chunk_ids = Column(JSON, nullable=True)     # list[str]

    faithfulness_score = Column(Float, nullable=False)
    faithfulness_reasoning = Column(Text, nullable=True)
    answer_relevance_score = Column(Float, nullable=False)
    answer_relevance_reasoning = Column(Text, nullable=True)
    context_relevance_score = Column(Float, nullable=False)
    context_relevance_reasoning = Column(Text, nullable=True)
    citation_accuracy_score = Column(Float, nullable=False)
    citation_accuracy_reasoning = Column(Text, nullable=True)

    overall_score = Column(Float, nullable=False)

    # Wall-clock time for the RAG pipeline only. Judge latency is recorded
    # separately so retrieval/generation cost is not conflated with scoring.
    pipeline_latency_ms = Column(Float, nullable=True)
    judge_latency_ms = Column(Float, nullable=True)

    # Relevance labels from the benchmark dataset and the retrieval metrics
    # they produce. All null for an unlabelled question.
    source_file = Column(String, nullable=True)
    relevant_pages = Column(JSON, nullable=True)      # list[int]
    retrieved_pages = Column(JSON, nullable=True)     # list[int | None], ranked
    precision_at_k = Column(Float, nullable=True)
    recall_at_k = Column(Float, nullable=True)
    reciprocal_rank = Column(Float, nullable=True)
    ndcg_at_k = Column(Float, nullable=True)

    run = relationship("EvalRun", back_populates="results")


# ---------------------------------------------------------------------------
# Retrieval-experiment tables (Week 4)
# ---------------------------------------------------------------------------
# An ExperimentConfig is a named, reusable set of strategies plus top_k. Each
# ExperimentRun records one strategy's ranked output for one query, so the
# Dense vs BM25 vs Hybrid comparison can be reviewed after the fact.


class ExperimentConfig(Base):
    __tablename__ = "experiment_configs"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False, unique=True, index=True)
    strategies = Column(JSON, nullable=False)  # list[str], RETRIEVER_REGISTRY keys
    top_k = Column(Integer, nullable=False, default=5)
    created_at = Column(DateTime(timezone=True), default=utcnow, index=True)

    runs = relationship("ExperimentRun", back_populates="config")


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    id = Column(String, primary_key=True)
    config_id = Column(
        String, ForeignKey("experiment_configs.id"), nullable=True, index=True
    )
    query = Column(Text, nullable=False)
    strategy = Column(String, nullable=False, index=True)
    top_k = Column(Integer, nullable=False)

    # completed | failed. A failed strategy keeps its error and no results,
    # rather than being dropped from the comparison.
    status = Column(String, nullable=False, default="completed")
    error_message = Column(Text, nullable=True)

    # Retrieval wall-clock time only; there is no generation in an experiment.
    latency_ms = Column(Float, nullable=True)
    # list[{"chunk_id", "file_name", "page_number", "score"}], best first.
    results = Column(JSON, nullable=True)

    created_at = Column(DateTime(timezone=True), default=utcnow, index=True)

    config = relationship("ExperimentConfig", back_populates="runs")


class RetrievalBenchmarkRun(Base):
    """One retrieval strategy scored over the shared benchmark (no generation).

    Strategies run together share a batch_id, so a comparison can be shown as
    it was measured. Metrics are at K over the labelled questions; latency is
    retrieval only, excluding the query embedding (reported per batch).
    """

    __tablename__ = "retrieval_benchmark_runs"

    id = Column(String, primary_key=True)
    batch_id = Column(String, nullable=False, index=True)
    dataset_name = Column(String, nullable=False)
    strategy = Column(String, nullable=False, index=True)
    k = Column(Integer, nullable=False)
    num_questions = Column(Integer, nullable=False)

    # completed | failed
    status = Column(String, nullable=False)
    error_message = Column(Text, nullable=True)

    avg_precision_at_k = Column(Float, nullable=True)
    avg_recall_at_k = Column(Float, nullable=True)
    avg_mrr = Column(Float, nullable=True)
    avg_ndcg_at_k = Column(Float, nullable=True)
    avg_latency_ms = Column(Float, nullable=True)

    by_type = Column(JSON, nullable=True)       # {type: {n, metric: mean}}
    per_question = Column(JSON, nullable=True)  # [{question, type, pages, metrics}]

    created_at = Column(DateTime(timezone=True), default=utcnow, index=True)
