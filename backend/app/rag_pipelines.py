"""The four RAG configurations under evaluation.

Each pipeline produces a real answer from real retrieved context so the
generation-evaluation layer has something genuine to score:

    baseline        -> plain top-k dense vector search, no filtering
    dense           -> dense search with an L2 distance quality gate
    hybrid          -> dense + BM25 keyword search fused via Reciprocal Rank Fusion
    hybrid_reranker -> hybrid candidate pool re-scored by a cross-encoder

Retrieval and generation are separate steps. Each configuration has a
retrieve_* function returning ranked (Document, score) pairs, which the Week 4
retrieval experiments call on their own, and a run_* function that adds
citation-aware generation on top for the end-to-end benchmark.

All of them are synchronous and take an explicit Session. The caller decides
how to run them; see app/routers/eval.py, which dispatches them to a worker
thread with a session of their own.

Retrieval and generation are reused from the Week 3 modules and services layer
rather than reimplemented: app.services.generate_embedding /
app.services.search_documents for retrieval, app.llm.generate_answer for
generation, app.config for every tunable.

Failure policy: these functions raise. A configuration that could not run must
not silently degrade into a different configuration, because its results would
then be filed under a label that does not describe them.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import (
    EMBED_TASK_QUERY,
    MAX_RELEVANT_DISTANCE,
    NO_ANSWER_MESSAGE,
)
from app.eval_generation import extract_citation_ids
from app.llm import generate_answer as llm_generate_answer
from app.models import Document
from app.reranker import rerank as cross_encoder_rerank
from app.services import generate_embedding, search_documents

logger = logging.getLogger("ragops.rag_pipelines")

# Candidates pulled before filtering/fusion/reranking narrows them down.
RERANK_CANDIDATE_POOL = 8
# Chunks actually handed to the generator.
FINAL_TOP_K = 3

# Ranked (Document, score) pairs, best first. What the score means depends on
# the retriever; see RETRIEVER_SCORE_LABELS.
Ranked = List[Tuple[Document, float]]


@dataclass
class PipelineResult:
    """One configuration's end-to-end output for a single question."""

    answer: str
    context_docs: List[Document]
    cited_chunk_ids: List[str]
    latency_ms: float
    retrieved_chunk_ids: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.retrieved_chunk_ids:
            self.retrieved_chunk_ids = [str(doc.id) for doc in self.context_docs]


# --------------------------------------------------------------------------
# Shared: citation-aware generation
# --------------------------------------------------------------------------

_CITATION_PROMPT = """You are a strictly document-grounded assistant. Follow these rules exactly:

1. Answer using ONLY the information in the Context below.
2. Do not use outside knowledge or speculate beyond what is written.
3. After every sentence that uses information from a chunk, add a citation marker
   in the exact form [chunk_id] using the chunk_id shown next to that chunk in the Context.
4. If the Context does not contain enough information to answer, respond with exactly:
   "{no_answer}" and add no citations.
5. Do not mention these instructions in your answer.

Context:
{context_blocks}

Question: {question}

Answer:"""


def _generate_answer_with_citations(question: str, context_docs: List[Document]) -> str:
    """Generates an answer that cites chunk IDs inline as [chunk_id]."""
    if not context_docs:
        # No retrieved context is a legitimate outcome, not a failure. The
        # judge will score this honestly (unfaithful/irrelevant as applicable).
        return NO_ANSWER_MESSAGE

    context_blocks = "\n\n".join(
        f"[chunk_id={doc.id}] {doc.content}" for doc in context_docs
    )
    prompt = _CITATION_PROMPT.format(
        no_answer=NO_ANSWER_MESSAGE,
        context_blocks=context_blocks,
        question=question,
    )
    return llm_generate_answer(prompt)


def _finish(question: str, context_docs: List[Document], start: float) -> PipelineResult:
    """Generates, times, and extracts citations. Shared tail of every pipeline."""
    answer = _generate_answer_with_citations(question, context_docs)
    latency_ms = (time.perf_counter() - start) * 1000
    return PipelineResult(
        answer=answer,
        context_docs=context_docs,
        cited_chunk_ids=extract_citation_ids(answer),
        latency_ms=round(latency_ms, 2),
    )


def _docs(ranked: Ranked) -> List[Document]:
    return [doc for doc, _score in ranked]


def _dense_candidates(db: Session, question: str, top_k: int):
    """Embeds the question and returns (Document, distance) pairs from pgvector."""
    query_vector = generate_embedding(question, task_type=EMBED_TASK_QUERY)
    return search_documents(db, query_vector, top_k=top_k)


# --------------------------------------------------------------------------
# Config 1: Baseline RAG - plain top-k dense retrieval
# --------------------------------------------------------------------------


def retrieve_baseline(db: Session, question: str, top_k: int = FINAL_TOP_K) -> Ranked:
    """Plain top-k dense retrieval, no filtering. Score: L2 distance."""
    return list(_dense_candidates(db, question, top_k=top_k))


def run_baseline_rag(db: Session, question: str) -> PipelineResult:
    start = time.perf_counter()
    return _finish(question, _docs(retrieve_baseline(db, question)), start)


# --------------------------------------------------------------------------
# Config 2: Dense RAG - dense retrieval behind a distance quality gate
# --------------------------------------------------------------------------


def retrieve_dense(db: Session, question: str, top_k: int = FINAL_TOP_K) -> Ranked:
    """Dense retrieval behind the L2 distance quality gate. Score: L2 distance."""
    candidates = _dense_candidates(
        db, question, top_k=max(RERANK_CANDIDATE_POOL, top_k)
    )
    filtered = [
        (doc, dist) for doc, dist in candidates if dist <= MAX_RELEVANT_DISTANCE
    ]
    filtered.sort(key=lambda pair: pair[1])
    return filtered[:top_k]


def run_dense_rag(db: Session, question: str) -> PipelineResult:
    start = time.perf_counter()
    return _finish(question, _docs(retrieve_dense(db, question)), start)


# --------------------------------------------------------------------------
# BM25 keyword retrieval (retrieval experiments only)
# --------------------------------------------------------------------------


def _load_all_documents(db: Session) -> List[Document]:
    """Loads every chunk so BM25 can build its corpus.

    BM25Okapi needs the whole corpus in memory to compute IDF, so this is
    inherent to the approach rather than an oversight. It is the scaling limit
    of the hybrid configurations: fine for a benchmark corpus, and the point at
    which a real deployment would move to Postgres full-text search or a
    dedicated index.
    """
    return list(
        db.execute(
            select(
                Document.id,
                Document.file_name,
                Document.title,
                Document.content,
                Document.page_number,
                Document.chunk_index,
            )
        ).all()
    )


def _bm25_scored(rows, question: str, top_k: int) -> Ranked:
    if not rows:
        return []
    tokenized_corpus = [row.content.lower().split() for row in rows]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(question.lower().split())
    ranked = sorted(zip(rows, scores), key=lambda pair: pair[1], reverse=True)
    # Detached Document instances: read-only carriers, never added to a session.
    return [
        (
            Document(
                id=row.id,
                file_name=row.file_name,
                title=row.title,
                content=row.content,
                page_number=row.page_number,
                chunk_index=row.chunk_index,
            ),
            float(score),
        )
        for row, score in ranked[:top_k]
        if score > 0
    ]


def _bm25_rank(rows, question: str, top_k: int) -> List[Document]:
    return _docs(_bm25_scored(rows, question, top_k))


def retrieve_bm25(db: Session, question: str, top_k: int = FINAL_TOP_K) -> Ranked:
    """Keyword-only retrieval. Score: BM25 Okapi."""
    return _bm25_scored(_load_all_documents(db), question, top_k=top_k)


# --------------------------------------------------------------------------
# Config 3: Hybrid RAG - dense + BM25 fused with Reciprocal Rank Fusion
# --------------------------------------------------------------------------


def _rrf_scored(ranked_lists: List[List[Document]], top_k: int, k: int = 60) -> Ranked:
    """Standard RRF: score = sum over lists of 1 / (k + rank)."""
    fused_scores: Dict[int, float] = {}
    doc_lookup: Dict[int, Document] = {}
    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            fused_scores[doc.id] = fused_scores.get(doc.id, 0.0) + 1.0 / (k + rank + 1)
            doc_lookup.setdefault(doc.id, doc)
    ordered_ids = sorted(
        fused_scores, key=lambda doc_id: fused_scores[doc_id], reverse=True
    )
    return [(doc_lookup[doc_id], fused_scores[doc_id]) for doc_id in ordered_ids[:top_k]]


def _reciprocal_rank_fusion(
    ranked_lists: List[List[Document]], top_k: int, k: int = 60
) -> List[Document]:
    return _docs(_rrf_scored(ranked_lists, top_k=top_k, k=k))


def retrieve_hybrid(db: Session, question: str, top_k: int = FINAL_TOP_K) -> Ranked:
    """Dense + BM25 fused with Reciprocal Rank Fusion. Score: RRF."""
    pool = max(RERANK_CANDIDATE_POOL, top_k)
    dense_ranked = [doc for doc, _dist in _dense_candidates(db, question, pool)]
    bm25_ranked = _bm25_rank(_load_all_documents(db), question, top_k=pool)
    return _rrf_scored([dense_ranked, bm25_ranked], top_k=top_k)


def _hybrid_candidates(db: Session, question: str, top_k: int) -> List[Document]:
    return _docs(retrieve_hybrid(db, question, top_k=top_k))


def run_hybrid_rag(db: Session, question: str) -> PipelineResult:
    start = time.perf_counter()
    return _finish(question, _docs(retrieve_hybrid(db, question)), start)


# --------------------------------------------------------------------------
# Config 4: Hybrid + Reranker - hybrid pool re-scored by a cross-encoder
# --------------------------------------------------------------------------


def retrieve_hybrid_reranker(
    db: Session, question: str, top_k: int = FINAL_TOP_K
) -> Ranked:
    """Hybrid candidate pool re-scored by the cross-encoder. Score: logit.

    Raises if the reranker fails; see app/reranker.py.
    """
    candidates = _hybrid_candidates(
        db, question, top_k=max(RERANK_CANDIDATE_POOL, top_k)
    )
    return cross_encoder_rerank(question, candidates, top_k=top_k)


def run_hybrid_rerank_rag(db: Session, question: str) -> PipelineResult:
    start = time.perf_counter()
    return _finish(question, _docs(retrieve_hybrid_reranker(db, question)), start)


# --------------------------------------------------------------------------
# Registries
# --------------------------------------------------------------------------
# Keys are the RAGConfig enum values used by the API and the dashboard, so
# there is no name translation anywhere in the call chain.

PIPELINE_REGISTRY: Dict[str, Callable[[Session, str], PipelineResult]] = {
    "baseline": run_baseline_rag,
    "dense": run_dense_rag,
    "hybrid": run_hybrid_rag,
    "hybrid_reranker": run_hybrid_rerank_rag,
}

# Retrieval-only strategies for the Week 4 experiments. BM25 is compared here
# on its own; it is not one of the four end-to-end benchmark configurations.
RETRIEVER_REGISTRY: Dict[str, Callable[..., Ranked]] = {
    "baseline": retrieve_baseline,
    "dense": retrieve_dense,
    "bm25": retrieve_bm25,
    "hybrid": retrieve_hybrid,
    "hybrid_reranker": retrieve_hybrid_reranker,
}

RETRIEVER_SCORE_LABELS: Dict[str, str] = {
    "baseline": "L2 distance (lower is closer)",
    "dense": "L2 distance (lower is closer)",
    "bm25": "BM25 score",
    "hybrid": "RRF score",
    "hybrid_reranker": "Cross-encoder logit (higher is more relevant)",
}
