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
LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-3.6-flash")

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

# ---------------------------------------------------------------------------
# Answer constants
# ---------------------------------------------------------------------------
NO_ANSWER_MESSAGE: str = (
    "I cannot find relevant information in the provided documents."
)
