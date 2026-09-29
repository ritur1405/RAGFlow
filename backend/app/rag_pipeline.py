"""RAG pipeline orchestrator.

Connects every stage of the dense RAG pipeline into a single callable
function that Member B can invoke from the Query API:

    query → embedding → dense retrieval → top-k chunks
          → context construction → prompt → LLM → answer + sources

Usage::

    from app.rag_pipeline import query_rag

    result = query_rag(db, "What is the main topic?")
    print(result.answer)
    for src in result.sources:
        print(src.document_name, src.text_snippet)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from sqlalchemy.orm import Session

from app.config import NO_ANSWER_MESSAGE, TOP_K
from app.context import build_context, build_prompt
from app.llm import generate_answer
from app.retrieval import RetrievedChunk, retrieve_chunks

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------


@dataclass
class SourceChunk:
    """A citation attached to the generated answer.

    Every field maps directly to data stored during ingestion — nothing
    is fabricated.
    """

    chunk_id: int
    document_name: Optional[str]
    text: str
    page_number: Optional[int]
    chunk_index: Optional[int]
    score: float  # L2 distance (lower = more similar)


@dataclass
class RAGResult:
    """The complete output of a RAG query.

    Attributes:
        answer:               The LLM-generated answer grounded in context.
        sources:              The actual chunks used to build that context.
        query:                The original user question (echo).
        num_chunks_retrieved: How many chunks passed the distance threshold.
    """

    answer: str
    sources: list[SourceChunk] = field(default_factory=list)
    query: str = ""
    num_chunks_retrieved: int = 0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def query_rag(
    db: Session,
    question: str,
    top_k: int | None = None,
) -> RAGResult:
    """Executes the full dense RAG pipeline and returns an answer with sources.

    This is the **single entry point** that the Query API should call.

    Args:
        db:       Active SQLAlchemy database session.
        question: The user's natural-language question.
        top_k:    Override the default number of retrieved chunks.

    Returns:
        A :class:`RAGResult` containing the generated answer and the
        source chunks that were used to produce it.

    Raises:
        ValueError: If *question* is empty or whitespace-only.
    """
    # --- Validate input ---------------------------------------------------
    if not question or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    question = question.strip()
    effective_top_k = top_k if top_k is not None else TOP_K

    # --- 1. Dense retrieval -----------------------------------------------
    try:
        chunks: list[RetrievedChunk] = retrieve_chunks(
            db, question, top_k=effective_top_k
        )
    except (RuntimeError, ValueError) as exc:
        logger.error("Retrieval failed for query %r: %s", question, exc)
        return RAGResult(
            answer=f"Retrieval error: {exc}",
            query=question,
        )

    # --- 2. Handle no results ---------------------------------------------
    if not chunks:
        logger.info("No relevant chunks found for query: %r", question)
        return RAGResult(
            answer=NO_ANSWER_MESSAGE,
            query=question,
            num_chunks_retrieved=0,
        )

    # --- 3. Context construction ------------------------------------------
    context = build_context(chunks)

    # --- 4. Prompt construction -------------------------------------------
    prompt = build_prompt(context, question)

    # --- 5. LLM generation ------------------------------------------------
    try:
        answer = generate_answer(prompt)
    except (RuntimeError, ValueError) as exc:
        logger.error("LLM generation failed for query %r: %s", question, exc)
        return RAGResult(
            answer=f"Generation error: {exc}",
            sources=_chunks_to_sources(chunks),
            query=question,
            num_chunks_retrieved=len(chunks),
        )

    # --- 6. Assemble result with citations --------------------------------
    return RAGResult(
        answer=answer,
        sources=_chunks_to_sources(chunks),
        query=question,
        num_chunks_retrieved=len(chunks),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunks_to_sources(chunks: list[RetrievedChunk]) -> list[SourceChunk]:
    """Converts retrieved chunks into citation objects.

    Each :class:`SourceChunk` maps 1-to-1 to a real chunk in PostgreSQL —
    no information is fabricated.
    """
    return [
        SourceChunk(
            chunk_id=c.chunk_id,
            document_name=c.document_name,
            text=c.content,
            page_number=c.page_number,
            chunk_index=c.chunk_index,
            score=c.similarity_score,
        )
        for c in chunks
    ]
