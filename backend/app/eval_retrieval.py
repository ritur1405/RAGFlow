"""Retrieval-quality metrics: Recall@K, Precision@K, MRR and nDCG@K.

These score the ranked chunks a configuration hands to the generator, against
page-level relevance labels in the benchmark dataset. A chunk is relevant when
it comes from a labelled page of the labelled file. Page labels are used
instead of chunk IDs because chunk IDs change every time a document is
re-ingested, while page numbers do not.

Definitions, for the top k ranked chunks:

    Precision@k  relevant chunks in the top k / k
    Recall@k     labelled pages hit by the top k / min(k, labelled pages)
                 (the capped recall of BEIR, Thakur et al. 2021, Appendix G,
                 so a question with many relevant pages is not unreachable)
    MRR          1 / rank of the first relevant chunk, 0 if there is none
                 (averaged over questions by the caller)
    nDCG@k       binary-gain DCG / ideal DCG. Every labelled page yields more
                 than k chunks, so the ideal ranking has k relevant chunks.

Pure functions with no I/O; app/routers/eval.py applies them per question.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from app.models import Document


@dataclass
class RetrievalScores:
    k: int
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float


def _same_file(doc_file: Optional[str], source_file: Optional[str]) -> bool:
    # No source_file in the label means the corpus is a single document.
    if not source_file:
        return True
    return (doc_file or "").casefold() == source_file.casefold()


def relevance(
    docs: Sequence[Document], source_file: Optional[str], relevant_pages: Iterable[int]
) -> list[bool]:
    """One flag per ranked chunk: is it from a labelled page of the labelled file?"""
    pages = set(relevant_pages)
    return [
        doc.page_number in pages and _same_file(doc.file_name, source_file) for doc in docs
    ]


def precision_at_k(rels: Sequence[bool], k: int) -> float:
    return sum(rels[:k]) / k


def recall_at_k(hit_pages: Iterable[int], relevant_pages: Iterable[int], k: int) -> float:
    relevant = set(relevant_pages)
    if not relevant:
        return 0.0
    return len(set(hit_pages) & relevant) / min(k, len(relevant))


def reciprocal_rank(rels: Sequence[bool]) -> float:
    for rank, rel in enumerate(rels, start=1):
        if rel:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(rels: Sequence[bool], k: int, num_relevant: Optional[int] = None) -> float:
    dcg = sum(1.0 / math.log2(rank + 1) for rank, rel in enumerate(rels[:k], start=1) if rel)
    ideal_hits = min(k, num_relevant) if num_relevant is not None else k
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def score_retrieval(
    docs: Sequence[Document],
    source_file: Optional[str],
    relevant_pages: Sequence[int],
    k: int,
) -> Optional[RetrievalScores]:
    """All four metrics for one question, or None when it has no labels."""
    if not relevant_pages:
        return None
    ranked = list(docs)[:k]
    rels = relevance(ranked, source_file, relevant_pages)
    hit_pages = [doc.page_number for doc, rel in zip(ranked, rels) if rel]
    return RetrievalScores(
        k=k,
        precision_at_k=precision_at_k(rels, k),
        recall_at_k=recall_at_k(hit_pages, relevant_pages, k),
        reciprocal_rank=reciprocal_rank(rels),
        ndcg_at_k=ndcg_at_k(rels, k),
    )
