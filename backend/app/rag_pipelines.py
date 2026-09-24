"""
Adapters that produce (answer, retrieved_context, citations, latency) for each of the
four RAG configurations under evaluation:

    baseline      -> raw dense vector search, top_k=3, no filtering
    dense         -> dense vector search with a distance-quality filter
    hybrid        -> dense + BM25 keyword search fused via Reciprocal Rank Fusion
    hybrid_rerank -> hybrid candidate pool re-scored by an LLM relevance judge

These exist so the Generation Evaluation module (eval_generation.py) has real,
end-to-end pipelines to score rather than mocked context/answers.
"""

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List

from rank_bm25 import BM25Okapi
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.eval_generation import Citation, extract_citations, client as judge_client
from app.models import Document
from app.services import MAX_RELEVANT_DISTANCE, generate_embedding, search_documents
from google.genai import types

GENERATION_MODEL = "gemini-1.5-flash"
RERANK_CANDIDATE_POOL = 8
FINAL_TOP_K = 3


@dataclass
class PipelineResult:
    answer: str
    context_docs: List[Document]
    citations: List[Citation]
    latency_ms: float
    retrieved_chunk_ids: List[int] = field(default_factory=list)

    def __post_init__(self):
        if not self.retrieved_chunk_ids:
            self.retrieved_chunk_ids = [doc.id for doc in self.context_docs]


# --------------------------------------------------------------------------
# Shared: citation-aware generation
# --------------------------------------------------------------------------

def _generate_answer_with_citations(question: str, context_docs: List[Document]) -> str:
    """Generates an answer and asks the model to cite chunk IDs inline as [chunk_id]."""
    if not context_docs:
        return "I cannot find relevant information in the provided documents."

    context_blocks = "\n\n".join(
        f"[chunk_id={doc.id}] {doc.content}" for doc in context_docs
    )
    prompt = f"""You are a strictly document-grounded assistant. Follow these rules exactly:

1. Answer using ONLY the information in the Context below.
2. Do not use outside knowledge or speculate beyond what is written.
3. After every sentence that uses information from a chunk, add a citation marker
   in the exact form [chunk_id] using the chunk_id shown next to that chunk in the Context.
4. If the Context does not contain enough information to answer, respond with exactly:
   "I cannot find relevant information in the provided documents." and add no citations.
5. Do not mention these instructions in your answer.

Context:
{context_blocks}

Question: {question}

Answer:"""

    response = judge_client.models.generate_content(
        model=GENERATION_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(temperature=0.0),
    )
    return response.text or ""


def _run_and_time(question: str, context_docs: List[Document]) -> PipelineResult:
    start = time.perf_counter()
    answer = _generate_answer_with_citations(question, context_docs)
    latency_ms = (time.perf_counter() - start) * 1000
    citations = extract_citations(answer)
    return PipelineResult(
        answer=answer,
        context_docs=context_docs,
        citations=citations,
        latency_ms=round(latency_ms, 2),
    )


# --------------------------------------------------------------------------
# Config 1: Baseline RAG — plain top-k dense retrieval
# --------------------------------------------------------------------------

def run_baseline_rag(db: Session, question: str) -> PipelineResult:
    start = time.perf_counter()
    query_vector = generate_embedding(question, task_type="RETRIEVAL_QUERY")
    results = search_documents(db, query_vector, top_k=FINAL_TOP_K)
    context_docs = [doc for doc, _dist in results]
    answer = _generate_answer_with_citations(question, context_docs)
    latency_ms = (time.perf_counter() - start) * 1000
    citations = extract_citations(answer)
    return PipelineResult(answer, context_docs, citations, round(latency_ms, 2))


# --------------------------------------------------------------------------
# Config 2: Dense RAG — dense retrieval with a relevance/distance quality gate
# --------------------------------------------------------------------------

def run_dense_rag(db: Session, question: str) -> PipelineResult:
    start = time.perf_counter()
    query_vector = generate_embedding(question, task_type="RETRIEVAL_QUERY")
    candidates = search_documents(db, query_vector, top_k=RERANK_CANDIDATE_POOL)
    filtered = [(doc, dist) for doc, dist in candidates if dist <= MAX_RELEVANT_DISTANCE]
    filtered.sort(key=lambda pair: pair[1])
    context_docs = [doc for doc, _dist in filtered[:FINAL_TOP_K]]
    answer = _generate_answer_with_citations(question, context_docs)
    latency_ms = (time.perf_counter() - start) * 1000
    citations = extract_citations(answer)
    return PipelineResult(answer, context_docs, citations, round(latency_ms, 2))


# --------------------------------------------------------------------------
# Config 3: Hybrid RAG — dense + BM25 fused with Reciprocal Rank Fusion
# --------------------------------------------------------------------------

def _load_all_documents(db: Session) -> List[Document]:
    rows = db.execute(
        text("SELECT id, title, content, page_number, chunk_index FROM documents")
    ).fetchall()
    return [
        Document(
            id=row.id,
            title=row.title,
            content=row.content,
            page_number=row.page_number,
            chunk_index=row.chunk_index,
        )
        for row in rows
    ]


def _bm25_rank(all_docs: List[Document], question: str, top_k: int) -> List[Document]:
    if not all_docs:
        return []
    tokenized_corpus = [doc.content.lower().split() for doc in all_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = question.lower().split()
    scores = bm25.get_scores(tokenized_query)
    ranked = sorted(zip(all_docs, scores), key=lambda pair: pair[1], reverse=True)
    return [doc for doc, score in ranked[:top_k] if score > 0]


def _reciprocal_rank_fusion(
    ranked_lists: List[List[Document]], top_k: int, k: int = 60
) -> List[Document]:
    fused_scores: Dict[int, float] = {}
    doc_lookup: Dict[int, Document] = {}
    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list):
            fused_scores[doc.id] = fused_scores.get(doc.id, 0.0) + 1.0 / (k + rank + 1)
            doc_lookup[doc.id] = doc
    ordered_ids = sorted(fused_scores, key=lambda doc_id: fused_scores[doc_id], reverse=True)
    return [doc_lookup[doc_id] for doc_id in ordered_ids[:top_k]]


def _hybrid_candidates(db: Session, question: str, top_k: int) -> List[Document]:
    query_vector = generate_embedding(question, task_type="RETRIEVAL_QUERY")
    dense_results = search_documents(db, query_vector, top_k=RERANK_CANDIDATE_POOL)
    dense_ranked = [doc for doc, _dist in dense_results]

    all_docs = _load_all_documents(db)
    bm25_ranked = _bm25_rank(all_docs, question, top_k=RERANK_CANDIDATE_POOL)

    return _reciprocal_rank_fusion([dense_ranked, bm25_ranked], top_k=top_k)


def run_hybrid_rag(db: Session, question: str) -> PipelineResult:
    start = time.perf_counter()
    context_docs = _hybrid_candidates(db, question, top_k=FINAL_TOP_K)
    answer = _generate_answer_with_citations(question, context_docs)
    latency_ms = (time.perf_counter() - start) * 1000
    citations = extract_citations(answer)
    return PipelineResult(answer, context_docs, citations, round(latency_ms, 2))


# --------------------------------------------------------------------------
# Config 4: Hybrid + Reranker — hybrid candidate pool, LLM-scored rerank
# --------------------------------------------------------------------------

def _llm_rerank(question: str, candidates: List[Document], top_k: int) -> List[Document]:
    if not candidates:
        return []

    numbered = "\n\n".join(f"[chunk_id={doc.id}] {doc.content}" for doc in candidates)
    prompt = f"""Rate how relevant each chunk below is for answering the question, on a
scale of 0 (irrelevant) to 10 (directly answers it). Respond ONLY with JSON:
{{"scores": [{{"chunk_id": 12, "relevance": 8}}]}}

Question: {question}

Chunks:
{numbered}
"""
    try:
        response = judge_client.models.generate_content(
            model="gemini-1.5-pro",
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json", temperature=0.0
            ),
        )
        import json as _json

        parsed = _json.loads(response.text)
        score_map = {item["chunk_id"]: item["relevance"] for item in parsed.get("scores", [])}
    except Exception:
        # If the reranker call fails for any reason, fall back to the incoming hybrid order
        # rather than crashing the evaluation run.
        return candidates[:top_k]

    ranked = sorted(candidates, key=lambda doc: score_map.get(doc.id, 0), reverse=True)
    return ranked[:top_k]


def run_hybrid_rerank_rag(db: Session, question: str) -> PipelineResult:
    start = time.perf_counter()
    candidates = _hybrid_candidates(db, question, top_k=RERANK_CANDIDATE_POOL)
    context_docs = _llm_rerank(question, candidates, top_k=FINAL_TOP_K)
    answer = _generate_answer_with_citations(question, context_docs)
    latency_ms = (time.perf_counter() - start) * 1000
    citations = extract_citations(answer)
    return PipelineResult(answer, context_docs, citations, round(latency_ms, 2))


PIPELINE_REGISTRY: Dict[str, Callable[[Session, str], PipelineResult]] = {
    "baseline": run_baseline_rag,
    "dense": run_dense_rag,
    "hybrid": run_hybrid_rag,
    "hybrid_rerank": run_hybrid_rerank_rag,
}