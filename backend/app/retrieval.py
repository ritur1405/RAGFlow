"""Dense retrieval module.

Converts a user query into an embedding, performs pgvector similarity
search, and returns structured results with metadata for citation
generation.

This module reuses the existing embedding and search functions from
services.py rather than reimplementing them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sqlalchemy.orm import Session

from app.config import (
    EMBED_TASK_QUERY,
    MAX_RELEVANT_DISTANCE,
    TOP_K,
)
from app.services import generate_embedding, search_documents


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    """A single chunk retrieved from the vector store.

    Attributes:
        chunk_id:         Primary key of the document row in PostgreSQL.
        document_name:    Original filename (e.g. ``sample.pdf``), if available.
        content:          The chunk text stored during ingestion.
        page_number:      1-indexed page from the source PDF, if available.
        chunk_index:      Global chunk position across the entire PDF.
        similarity_score: L2 distance from pgvector (lower = more similar).
    """

    chunk_id: int
    document_name: Optional[str]
    content: str
    page_number: Optional[int]
    chunk_index: Optional[int]
    similarity_score: float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def retrieve_chunks(
    db: Session,
    query: str,
    top_k: int | None = None,
    max_distance: float | None = None,
) -> list[RetrievedChunk]:
    """Full dense retrieval pipeline: query → embedding → pgvector search.

    Args:
        db:           Active SQLAlchemy session.
        query:        The user's natural-language question.
        top_k:        Number of chunks to retrieve.  Falls back to the
                      ``RAG_TOP_K`` environment variable / config default.
        max_distance: L2 distance threshold.  Chunks above this value are
                      discarded.  Falls back to ``RAG_MAX_DISTANCE`` / config.

    Returns:
        A list of :class:`RetrievedChunk` objects ordered by ascending
        distance (most similar first), filtered by the distance threshold.

    Raises:
        ValueError: If *query* is empty or whitespace-only.
        RuntimeError: If embedding generation fails.
    """
    if not query or not query.strip():
        raise ValueError("Query must be a non-empty string.")

    effective_top_k = top_k if top_k is not None else TOP_K
    effective_max_distance = (
        max_distance if max_distance is not None else MAX_RELEVANT_DISTANCE
    )

    # 1. Generate query embedding using the SAME model as ingestion,
    #    but with the RETRIEVAL_QUERY task type.
    try:
        query_vector = generate_embedding(query, task_type=EMBED_TASK_QUERY)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to generate query embedding: {exc}"
        ) from exc

    # 2. Search pgvector for nearest neighbours.
    try:
        scored_docs = search_documents(db, query_vector, top_k=effective_top_k)
    except Exception as exc:
        raise RuntimeError(
            f"Vector search failed: {exc}"
        ) from exc

    # 3. Convert (Document, distance) pairs into RetrievedChunk objects,
    #    filtering out anything beyond the relevance threshold.
    results: list[RetrievedChunk] = []
    for doc, distance in scored_docs:
        if distance > effective_max_distance:
            continue
        results.append(
            RetrievedChunk(
                chunk_id=doc.id,
                document_name=getattr(doc, "file_name", None),
                content=doc.content,
                page_number=doc.page_number,
                chunk_index=doc.chunk_index,
                similarity_score=round(distance, 6),
            )
        )

    return results
