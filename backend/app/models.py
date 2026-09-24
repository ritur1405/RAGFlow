from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    Float,
    DateTime,
    ForeignKey,
    JSON,
    func,
)
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from app.database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, nullable=True)     # Stores original filename (e.g. sample.pdf)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=True)  # Page from PDF
    chunk_index = Column(Integer, nullable=True)  # Position index of chunk
    embedding = Column(Vector(768))                # pgvector embedding


class EvaluationRun(Base):
    """A single evaluation job: one benchmark dataset run across one or more RAG configs."""

    __tablename__ = "evaluation_runs"

    id = Column(Integer, primary_key=True, index=True)
    run_name = Column(String, nullable=False)
    rag_configs = Column(JSON, nullable=False)          # e.g. ["baseline", "dense", "hybrid", "hybrid_rerank"]
    benchmark_size = Column(Integer, nullable=False, default=0)
    status = Column(String, nullable=False, default="pending")  # pending | running | completed | failed
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)

    results = relationship(
        "EvaluationResult", back_populates="run", cascade="all, delete-orphan"
    )


class EvaluationResult(Base):
    """Per-question, per-RAG-config generation evaluation result."""

    __tablename__ = "evaluation_results"

    id = Column(Integer, primary_key=True, index=True)
    run_id = Column(Integer, ForeignKey("evaluation_runs.id"), nullable=False, index=True)
    rag_config = Column(String, nullable=False, index=True)  # baseline | dense | hybrid | hybrid_rerank

    question = Column(Text, nullable=False)
    reference_answer = Column(Text, nullable=True)
    generated_answer = Column(Text, nullable=False)

    retrieved_chunk_ids = Column(JSON, nullable=False, default=list)
    context_snapshot = Column(JSON, nullable=False, default=list)  # [{chunk_id, title, content}]
    citations = Column(JSON, nullable=False, default=list)         # [{chunk_id, sentence}]

    faithfulness_score = Column(Float, nullable=False)
    faithfulness_reasoning = Column(Text, nullable=False)

    answer_relevance_score = Column(Float, nullable=False)
    answer_relevance_reasoning = Column(Text, nullable=False)

    context_relevance_score = Column(Float, nullable=False)
    context_relevance_reasoning = Column(Text, nullable=False)

    citation_accuracy_score = Column(Float, nullable=False)
    citation_accuracy_reasoning = Column(Text, nullable=False)

    overall_score = Column(Float, nullable=False)
    latency_ms = Column(Float, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    run = relationship("EvaluationRun", back_populates="results")