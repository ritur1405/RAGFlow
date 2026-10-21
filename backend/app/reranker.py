"""Cross-encoder reranker for the hybrid_reranker configuration.

A cross-encoder reads the question and one chunk together and outputs a
relevance logit, so it judges relevance more precisely than the bi-encoder
embeddings used for first-stage retrieval, at the cost of one forward pass per
candidate. That is why it only re-scores the small hybrid candidate pool.

The model runs locally through fastembed's ONNX runtime: no PyTorch, no API
call and no per-query cost, and it is deterministic, so benchmark runs are
repeatable. The Docker image downloads it at build time (see the Dockerfile),
and the first call in any other environment downloads it once.

Failure policy matches app/rag_pipelines.py: errors raise. Returning the
incoming order would make the configuration silently identical to plain
hybrid while still being recorded as reranked.
"""

from __future__ import annotations

import logging
import threading
from typing import List, Tuple

from app.config import RERANKER_MODEL
from app.models import Document

logger = logging.getLogger("ragops.reranker")

_encoder = None
_encoder_lock = threading.Lock()


def _get_encoder():
    """Loads the model once per process. Pipelines run in worker threads."""
    global _encoder
    if _encoder is None:
        with _encoder_lock:
            if _encoder is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                logger.info("Loading cross-encoder %s", RERANKER_MODEL)
                _encoder = TextCrossEncoder(model_name=RERANKER_MODEL)
    return _encoder


def warm_up() -> None:
    """Loads the model ahead of the first query, so no request pays for it.

    Without this the first reranked query also measures model loading
    (several seconds), which would distort its latency in a benchmark.
    Failure is logged, not raised: the app can still serve every other
    configuration, and a reranked query will then raise as usual.
    """
    try:
        _get_encoder()
    except Exception:
        logger.exception("Could not pre-load the cross-encoder reranker")


def rerank(question: str, candidates: List[Document], top_k: int) -> List[Tuple[Document, float]]:
    """Returns the top_k candidates as (Document, logit) pairs, most relevant first."""
    if not candidates:
        return []
    try:
        scores = list(_get_encoder().rerank(question, [doc.content for doc in candidates]))
    except Exception as exc:
        logger.exception("Cross-encoder reranker failed for question=%r", question)
        raise RuntimeError(f"Cross-encoder reranker failed: {exc}") from exc

    if len(scores) != len(candidates):
        raise RuntimeError(
            f"Cross-encoder reranker failed: {len(scores)} scores for {len(candidates)} candidates"
        )

    scored = [(doc, float(score)) for doc, score in zip(candidates, scores)]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]
