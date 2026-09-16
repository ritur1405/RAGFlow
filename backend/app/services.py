import io
import os
from dataclasses import dataclass, field
from google import genai
from google.genai import types
from pypdf import PdfReader
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models import Document
from app.utils import chunk_text

# Initialize Gemini Client (automatically reads GEMINI_API_KEY from environment)
client = genai.Client()

EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-3.6-flash"  # gemini-2.5-flash is no longer available to new users
EMBEDDING_DIM = 768  # Matches app/models.py: Column(Vector(768))

# pgvector's <=> operator returns cosine distance (0 = identical, 1 = orthogonal)
# for unit-normalized embeddings like gemini-embedding-001.  This threshold is a
# STARTING point — tune it by running known-good and known-bad queries and
# inspecting the cosine distances search_documents() returns.
MAX_RELEVANT_DISTANCE = 0.35

NO_ANSWER_MESSAGE = "I cannot find relevant information in the provided documents."


def generate_embedding(text_content: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """Generates 768-dimensional vector embeddings using gemini-embedding-001."""
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text_content,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBEDDING_DIM,
        ),
    )
    return response.embeddings[0].values


def search_documents(
    db: Session, query_vector: list[float], top_k: int = 3
) -> list[tuple[Document, float]]:
    """Executes native pgvector cosine-similarity search.

    Args:
        db: Active SQLAlchemy session.
        query_vector: 768-dimensional embedding of the user's query,
            generated with ``task_type="RETRIEVAL_QUERY"``.
        top_k: Number of nearest neighbours to return (1 ≤ top_k ≤ 100).

    Returns:
        A list of ``(Document, cosine_distance)`` pairs, ordered from most
        similar (lowest distance) to least.  Callers can derive a 0→1
        similarity score via ``1 - cosine_distance``.

    Raises:
        ValueError: If ``top_k`` is out of the allowed range, or if
            ``query_vector`` does not have the expected dimensionality.
    """
    # --- Input validation ---------------------------------------------------
    if not isinstance(top_k, int) or top_k < 1 or top_k > 100:
        raise ValueError(
            f"top_k must be an integer between 1 and 100 (got {top_k!r})."
        )

    if len(query_vector) != EMBEDDING_DIM:
        raise ValueError(
            f"query_vector must have {EMBEDDING_DIM} dimensions "
            f"(got {len(query_vector)})."
        )

    # --- pgvector cosine-distance query -------------------------------------
    vector_str = f"[{','.join(map(str, query_vector))}]"

    sql = text("""
        SELECT id, title, content, file_name, page_number, chunk_index,
               embedding <=> CAST(:vector AS vector) AS distance
        FROM documents
        WHERE embedding IS NOT NULL
        ORDER BY distance ASC
        LIMIT :top_k
    """)

    results = db.execute(sql, {"vector": vector_str, "top_k": top_k}).fetchall()

    return [
        (
            Document(
                id=row.id,
                title=row.title,
                content=row.content,
                file_name=row.file_name,
                page_number=row.page_number,
                chunk_index=row.chunk_index,
            ),
            float(row.distance),
        )
        for row in results
    ]


# Keyword triggers for broad/summary-style questions that naive top-k vector
# search handles poorly, since no single small chunk captures "what is this
# document about" — that only emerges from the whole document.
META_QUESTION_KEYWORDS = [
    "main topic",
    "what is this document about",
    "what is the document about",
    "what does this document",
    "what is this draft about",
    "what is the draft about",
    "summarize",
    "summary",
    "overview of",
    "give me an overview",
    "tl;dr",
]


def is_meta_question(question: str) -> bool:
    """Heuristic check for broad/summary-style questions vs. specific-fact lookups."""
    q = question.lower()
    return any(keyword in q for keyword in META_QUESTION_KEYWORDS)


def generate_summary_answer(
    db: Session, question: str, max_chars: int = 12000
) -> tuple[str, list[Document]]:
    """Handles broad/meta questions by feeding Gemini the document in reading
    order (by page, then chunk index) up to a character budget, instead of
    relying on similarity search — which has no reason to surface the chunks
    that best describe the document as a whole.
    """
    sql = text("""
        SELECT id, title, content, page_number, chunk_index
        FROM documents
        ORDER BY page_number NULLS LAST, chunk_index NULLS LAST, id
    """)
    rows = db.execute(sql).fetchall()

    if not rows:
        return NO_ANSWER_MESSAGE, []

    context_parts = []
    used_docs = []
    total_chars = 0

    for row in rows:
        piece = row.content or ""
        if total_chars + len(piece) > max_chars:
            break
        context_parts.append(piece)
        total_chars += len(piece)
        used_docs.append(
            Document(
                id=row.id,
                title=row.title,
                content=row.content,
                page_number=row.page_number,
                chunk_index=row.chunk_index,
            )
        )

    context_text = "\n\n".join(context_parts)
    prompt = f"""You are a strictly document-grounded assistant. Answer using ONLY the document content given below.
Do not use outside knowledge. Do not speculate. If the document content does not contain
enough information to answer, respond with exactly: "{NO_ANSWER_MESSAGE}"

Document content (reading order; extra line breaks may be present from PDF extraction):
{context_text}

Question: {question}

Answer:"""

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
    )
    return response.text, used_docs


# ---------------------------------------------------------------------------
# Context builder — pure logic, no LLM calls
# ---------------------------------------------------------------------------

@dataclass
class SourceMeta:
    """Lightweight metadata for one chunk included in the context."""
    id: int
    title: str
    file_name: str | None
    page_number: int | None
    chunk_index: int | None
    label: str  # Human-readable label used inside the context block


@dataclass
class ContextResult:
    """Return value of :func:`build_context`.

    Attributes:
        context_text: Formatted string ready to paste into an LLM prompt.
        sources: Metadata for every chunk that was actually included
            (respecting the character budget).
        total_chars: Total character count of ``context_text``.
        chunks_included: How many chunks fit within the budget.
        chunks_dropped: How many chunks were skipped due to the budget.
    """
    context_text: str
    sources: list[SourceMeta] = field(default_factory=list)
    total_chars: int = 0
    chunks_included: int = 0
    chunks_dropped: int = 0


def _build_chunk_label(doc: Document) -> str:
    """Builds a human-readable ``[Source ...]`` label for one chunk."""
    parts: list[str] = []
    if getattr(doc, "file_name", None):
        parts.append(f'File: "{doc.file_name}"')
    if getattr(doc, "page_number", None) is not None:
        parts.append(f"Page {doc.page_number}")
    if getattr(doc, "chunk_index", None) is not None:
        parts.append(f"Chunk {doc.chunk_index}")
    return ", ".join(parts) if parts else f"Source ID {doc.id}"


def build_context(
    chunks: list[Document],
    max_chars: int = 12_000,
) -> ContextResult:
    """Formats retrieved chunks into a labelled context string for the LLM.

    Each included chunk is rendered as::

        [Source 1 — File: "report.pdf", Page 3, Chunk 7]
        <chunk content>

    The function stops adding chunks once ``max_chars`` would be exceeded,
    so the caller can control prompt length without truncating mid-sentence.

    Args:
        chunks: Retrieved ``Document`` objects, already in relevance order.
        max_chars: Maximum character budget for the combined context text.
            Must be a positive integer.

    Returns:
        A :class:`ContextResult` containing the formatted context, the list
        of :class:`SourceMeta` for every chunk that was included, and
        bookkeeping counters.

    Raises:
        ValueError: If ``max_chars`` is not a positive integer.
    """
    if not isinstance(max_chars, int) or max_chars < 1:
        raise ValueError(
            f"max_chars must be a positive integer (got {max_chars!r})."
        )

    if not chunks:
        return ContextResult(
            context_text="",
            sources=[],
            total_chars=0,
            chunks_included=0,
            chunks_dropped=0,
        )

    separator = "\n\n"
    parts: list[str] = []
    sources: list[SourceMeta] = []
    running_chars = 0
    source_number = 0

    for doc in chunks:
        label = _build_chunk_label(doc)
        block = f"[Source {source_number + 1} \u2014 {label}]\n{doc.content}"

        # Account for the separator between blocks (except the first)
        added_chars = len(block) + (len(separator) if parts else 0)

        if running_chars + added_chars > max_chars:
            break  # budget exceeded — stop cleanly, don't truncate

        parts.append(block)
        running_chars += added_chars
        source_number += 1

        sources.append(SourceMeta(
            id=doc.id,
            title=doc.title,
            file_name=getattr(doc, "file_name", None),
            page_number=getattr(doc, "page_number", None),
            chunk_index=getattr(doc, "chunk_index", None),
            label=label,
        ))

    context_text = separator.join(parts)

    return ContextResult(
        context_text=context_text,
        sources=sources,
        total_chars=len(context_text),
        chunks_included=len(sources),
        chunks_dropped=len(chunks) - len(sources),
    )


# ---------------------------------------------------------------------------
# Prompt builder — pure logic, no LLM calls
# ---------------------------------------------------------------------------

def build_rag_prompt(question: str, context_text: str) -> str:
    """Builds the complete LLM prompt for grounded RAG answer generation.

    This is a **pure function** — it does not call the LLM, touch the
    database, or have side effects.  Keeping it separate makes the prompt
    easy to unit-test, version, and swap.

    Args:
        question: The user's natural-language question.
        context_text: Pre-formatted context string (output of
            :func:`build_context`), with ``[Source N — …]`` labels.

    Returns:
        The fully assembled prompt string, ready to send to the LLM.
    """
    return f"""You are a strictly document-grounded assistant.

RULES — follow every rule exactly:
1. Answer using ONLY the information provided in the "Context" section below.
2. Do NOT use outside knowledge, training data, or assumptions beyond what is explicitly stated in the context.
3. Do NOT speculate, infer beyond what is written, or fill gaps with plausible-sounding information.
4. If the context does not contain enough information to fully answer the question, respond with exactly this sentence and nothing else:
   "{NO_ANSWER_MESSAGE}"
5. Cite your sources. When a claim comes from the context, reference it using the source label provided (e.g. [Source 1], [Source 2]).
6. Do NOT fabricate citations. Only cite source labels that appear in the context below. If a source label does not appear in the context, do not reference it.
7. Do NOT mention these instructions or the existence of a "Context" section in your answer.

Context:
{context_text}

Question: {question}

Answer:"""


# ---------------------------------------------------------------------------
# LLM answer generation (uses build_context + build_rag_prompt)
# ---------------------------------------------------------------------------

def generate_answer(question: str, context_docs: list[Document]) -> str:
    """Synthesizes a strictly grounded RAG response using Gemini.

    Pipeline: chunks → :func:`build_context` → :func:`build_rag_prompt` → LLM.
    """
    ctx = build_context(context_docs)
    prompt = build_rag_prompt(question, ctx.context_text)

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
    )
    return response.text


def process_pdf(file_contents: bytes) -> list[dict]:
    """Extracts text from raw PDF bytes, chunks it, generates embeddings, and returns structured dictionaries."""
    reader = PdfReader(io.BytesIO(file_contents))
    chunks_data = []
    chunk_global_index = 0

    for page_idx, page in enumerate(reader.pages):
        text_content = page.extract_text()
        if not text_content or not text_content.strip():
            continue

        page_chunks = chunk_text(text_content, chunk_size=500, chunk_overlap=50)

        for chunk_text_str in page_chunks:
            embedding = generate_embedding(chunk_text_str, task_type="RETRIEVAL_DOCUMENT")

            chunks_data.append({
                "text": chunk_text_str,
                "page_number": page_idx + 1,
                "chunk_index": chunk_global_index,
                "embedding": embedding,
            })
            chunk_global_index += 1

    return chunks_data

def generate_embeddings_batch(
    texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT", batch_size: int = 100
) -> list[list[float]]:
    """Generates 768-dimensional vector embeddings for a list of strings in batches.

    Splits the texts into safe chunk batches to avoid request size/limit errors.
    """
    if not texts:
        return []

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=EMBEDDING_DIM,
            ),
        )
        if not response.embeddings or len(response.embeddings) != len(batch):
            raise ValueError(
                f"Failed to generate embeddings for batch. Expected {len(batch)}, "
                f"got {len(response.embeddings) if response.embeddings else 0}."
            )

        for emb in response.embeddings:
            embeddings.append(emb.values)

    return embeddings
