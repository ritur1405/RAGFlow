"""Context and prompt construction for the RAG pipeline.

Converts retrieved chunks into a numbered, source-attributed context
block and wraps it in a prompt that instructs the LLM to stay grounded.

These are pure functions with no side effects, making them easy to test
and swap out.
"""

from __future__ import annotations

from app.config import NO_ANSWER_MESSAGE

# The retrieval module is imported only for the type hint; there is no
# circular dependency because context.py has no module-level side effects.
from app.retrieval import RetrievedChunk


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

# Kept as a module-level constant so it can be overridden or swapped easily.
# The ``{context}`` and ``{question}`` placeholders are filled at runtime.
PROMPT_TEMPLATE: str = """\
You are a strictly document-grounded assistant. Follow these rules exactly:

1. Answer using ONLY the information in the "Context" section below.
2. Do not use outside knowledge, training data, or assumptions beyond what is stated in the context.
3. Do not speculate, infer beyond what is written, or fill gaps with plausible-sounding information.
4. If the context does not contain enough information to answer the question, respond with \
exactly this sentence and nothing else: "{no_answer}"
5. When you use information from a specific source, reference it using its identifier \
(e.g. [Source 1], [Source 2]) so the user can trace your answer back to the original document.
6. Provide a concise, useful answer. Do not repeat the question or these instructions.

Context:
{context}

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Converts retrieved chunks into numbered, source-attributed context.

    Each chunk is rendered as::

        [Source 1 | document.pdf | Page 3 | Chunk 7]
        <chunk text>

    so the LLM (and ultimately the user) can trace every claim back to an
    actual ingested source.

    Args:
        chunks: Ordered list of retrieved chunks (most relevant first).

    Returns:
        A single string containing all sources, separated by blank lines.
        Returns an empty string if *chunks* is empty.
    """
    if not chunks:
        return ""

    parts: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        # Build a human-readable source label with available metadata.
        label_parts = [f"Source {idx}"]
        if chunk.document_name:
            label_parts.append(chunk.document_name)
        if chunk.page_number is not None:
            label_parts.append(f"Page {chunk.page_number}")
        if chunk.chunk_index is not None:
            label_parts.append(f"Chunk {chunk.chunk_index}")

        header = "[" + " | ".join(label_parts) + "]"
        parts.append(f"{header}\n{chunk.content}")

    return "\n\n".join(parts)


def build_prompt(context: str, question: str) -> str:
    """Constructs the full LLM prompt from context and question.

    Args:
        context:  The formatted context string from :func:`build_context`.
        question: The user's original question.

    Returns:
        A ready-to-send prompt string.
    """
    return PROMPT_TEMPLATE.format(
        context=context,
        question=question,
        no_answer=NO_ANSWER_MESSAGE,
    )
