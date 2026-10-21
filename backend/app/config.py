"""Centralized RAG configuration.

All tunable RAG parameters live here so they can be adjusted via
environment variables without hunting through multiple source files.
"""

import os


# ---------------------------------------------------------------------------
# Embedding configuration
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "768"))

# Task types used by Gemini's embed_content API to optimise the embedding
# for the intended use case.
EMBED_TASK_DOCUMENT: str = "RETRIEVAL_DOCUMENT"
EMBED_TASK_QUERY: str = "RETRIEVAL_QUERY"

# ---------------------------------------------------------------------------
# LLM configuration
# ---------------------------------------------------------------------------
# Defaults are chosen to fit the Gemini free tier (limits per model, per day):
# gemini-3.5-flash-lite allows 500 requests/day versus 20 for gemini-3.6-flash,
# enough for the Ask page and a full benchmark run without billing.
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")

# The evaluation judge (app/eval_generation.py). A different model family from
# the generator, so the judge is not grading its own model's answers, with its
# own free-tier quota (14,400 requests/day).
JUDGE_MODEL: str = os.getenv("GEMINI_JUDGE_MODEL", "gemma-4-31b-it")

# Total attempts per Gemini call on a per-minute rate limit or a transient
# server error (app/services.py, app/quota.py). Daily-quota errors are never
# retried.
GEMINI_RETRY_ATTEMPTS: int = int(os.getenv("GEMINI_RETRY_ATTEMPTS", "8"))

# ---------------------------------------------------------------------------
# Retrieval configuration
# ---------------------------------------------------------------------------
# Default number of chunks returned by dense retrieval.
TOP_K: int = int(os.getenv("RAG_TOP_K", "5"))

# pgvector's <-> operator returns L2 distance (lower = more similar).
# Chunks with distance above this threshold are considered irrelevant and
# will be excluded from the context sent to the LLM.  Tune this value
# against your own data — it depends on the embedding model and the
# typical document domain.
MAX_RELEVANT_DISTANCE: float = float(os.getenv("RAG_MAX_DISTANCE", "0.8"))

# Cross-encoder used by the hybrid_reranker configuration (app/reranker.py).
# Any reranker fastembed supports works, e.g. "BAAI/bge-reranker-base".
RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")

# ---------------------------------------------------------------------------
# Answer constants
# ---------------------------------------------------------------------------
NO_ANSWER_MESSAGE: str = (
    "I cannot find relevant information in the provided documents."
)
