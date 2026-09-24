from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

RagConfigName = Literal["baseline", "dense", "hybrid", "hybrid_rerank"]

ALL_RAG_CONFIGS: List[RagConfigName] = ["baseline", "dense", "hybrid", "hybrid_rerank"]


class BenchmarkItem(BaseModel):
    question: str
    reference_answer: Optional[str] = None


class EvalRunRequest(BaseModel):
    run_name: str = Field(..., min_length=1)
    benchmark: List[BenchmarkItem] = Field(..., min_length=1)
    rag_configs: List[RagConfigName] = Field(default_factory=lambda: list(ALL_RAG_CONFIGS))


class MetricScore(BaseModel):
    score: float
    reasoning: str


class CitationRecord(BaseModel):
    chunk_id: Optional[int]
    sentence: str


class ContextChunk(BaseModel):
    chunk_id: int
    title: str
    content: str


class EvalResultItem(BaseModel):
    id: int
    rag_config: RagConfigName
    question: str
    reference_answer: Optional[str]
    generated_answer: str
    retrieved_chunk_ids: List[int]
    context_snapshot: List[ContextChunk]
    citations: List[CitationRecord]
    faithfulness: MetricScore
    answer_relevance: MetricScore
    context_relevance: MetricScore
    citation_accuracy: MetricScore
    overall_score: float
    latency_ms: float


class EvalRunResponse(BaseModel):
    run_id: int
    run_name: str
    status: str
    rag_configs: List[str]
    benchmark_size: int
    created_at: datetime
    completed_at: Optional[datetime]
    results: List[EvalResultItem]


class EvalResultsResponse(BaseModel):
    run_id: int
    run_name: str
    status: str
    results: List[EvalResultItem]


class ConfigAggregate(BaseModel):
    rag_config: RagConfigName
    num_questions: int
    avg_faithfulness: float
    avg_answer_relevance: float
    avg_context_relevance: float
    avg_citation_accuracy: float
    avg_overall_score: float
    avg_latency_ms: float


class EvalCompareResponse(BaseModel):
    run_id: int
    run_name: str
    configs: List[ConfigAggregate]


class EvalRunSummary(BaseModel):
    run_id: int
    run_name: str
    status: str
    rag_configs: List[str]
    benchmark_size: int
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None