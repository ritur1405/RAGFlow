import io
import os
from typing import Generator

from google import genai
from google.genai import types
from pypdf import PdfReader
 week-3
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Document
from app.utils import chunk_text

# Initialize Gemini Client (reads GEMINI_API_KEY from environment)
client = genai.Client()

# Dynamic model selection from environment variables with defaults
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.6-flash")
EMBEDDING_DIM = 768

MAX_RELEVANT_DISTANCE = 0.5
NO_ANSWER_MESSAGE = "I cannot find relevant information in the provided documents."

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

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.models import Document
from app.utils import chunk_text

# Initialize Gemini Client (automatically reads GEMINI_API_KEY from environment)
client = genai.Client()

EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-3.6-flash"  # gemini-2.5-flash is no longer available to new users
EMBEDDING_DIM = 768  # Matches app/models.py: Column(Vector(768))
main

# pgvector's <-> operator returns L2 distance (lower = more similar) for a
# plain Vector column with no explicit ops class. This is a STARTING threshold,
# not a universal constant — tune it against your own data/embedding model by
# running a batch of known-good and known-bad queries and inspecting the
# distances search_documents() returns.
MAX_RELEVANT_DISTANCE = 0.8

week-3
def generate_embedding(text_content: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:

NO_ANSWER_MESSAGE = "I cannot find relevant information in the provided documents."


def generate_embedding(text_content: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """Generates 768-dimensional vector embeddings using gemini-embedding-001."""
 main
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text_content,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBEDDING_DIM,
        ),
    )
    return response.embeddings[0].values


def generate_embeddings_batch(
    texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT"
) -> list[list[float]]:
    if not texts:
        return []

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBEDDING_DIM,
        ),
    )
    return [e.values for e in response.embeddings]


def search_documents(
    db: Session, query_vector: list[float], top_k: int = 3
) -> list[tuple[Document, float]]:
 week-3
    distance_expr = Document.embedding.cosine_distance(query_vector)

    stmt = (
        select(Document, distance_expr.label("distance"))
        .where(Document.embedding.is_not(None))
        .order_by(distance_expr)
        .limit(top_k)
    )

    results = db.execute(stmt).all()
    return [(row[0], float(row[1])) for row in results]


def is_meta_question(question: str) -> bool:
    q = question.lower()
    return any(keyword in q for keyword in META_QUESTION_KEYWORDS)


def generate_summary_answer(
    db: Session, question: str, max_chars: int = 12000
) -> tuple[str, list[Document]]:
    stmt = (
        select(Document)
        .order_by(
            Document.page_number.is_(None), Document.page_number,
            Document.chunk_index.is_(None), Document.chunk_index,
            Document.id,
        )
    )
    rows = db.execute(stmt).scalars().all()

    if not rows:
        return NO_ANSWER_MESSAGE, []

    context_parts = []
    used_docs = []
    total_chars = 0

    for doc in rows:
        piece = doc.content or ""
        if total_chars + len(piece) > max_chars:
            break
        context_parts.append(f"[Source: {doc.file_name or doc.title}] {piece}")
        total_chars += len(piece)
        used_docs.append(doc)

    context_text = "\n\n".join(context_parts)
    prompt = f"""You are a strictly document-grounded assistant. Answer using ONLY the document content given below.
Do not use outside knowledge. Do not speculate. If the document content does not contain
enough information to answer, respond with exactly: "{NO_ANSWER_MESSAGE}"

Document content:
{context_text}

Question: {question}

Answer:"""

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
    )
    return response.text, used_docs


def _build_grounded_prompt(question: str, context_docs: list[Document]) -> str:
    context_blocks = [
        f"[Source: {doc.file_name or doc.title} | Page {doc.page_number}] {doc.content}"
        for doc in context_docs

    """Executes native pgvector similarity search (Supabase supports pgvector directly).

    Returns (Document, distance) pairs instead of bare Documents, so callers
    can apply a relevance threshold and fall back gracefully instead of
    always forcing an answer out of whatever happened to rank highest.
    """
    vector_str = f"[{','.join(map(str, query_vector))}]"

    sql = text("""
        SELECT id, title, content, page_number, chunk_index,
               embedding <-> CAST(:vector AS vector) AS distance
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
                page_number=row.page_number,
                chunk_index=row.chunk_index,
            ),
            float(row.distance),
        )
        for row in results
main
    ]
    context_text = "\n\n".join(context_blocks)

    return f"""You are a strictly document-grounded assistant. Follow these rules exactly:

 week-3

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


def generate_answer(question: str, context_chunks: list[str]) -> str:
    """Synthesizes a strictly grounded RAG response using Gemini based on retrieved chunks."""
    context_text = "\n\n".join(context_chunks)
    prompt = f"""You are a strictly document-grounded assistant. Follow these rules exactly:

 main
1. Answer using ONLY the information in the "Context" section below.
2. Do not use outside knowledge, training data, or assumptions beyond what is stated in the context.
3. Do not speculate, infer beyond what's written, or fill gaps with plausible-sounding information.
4. If the context does not contain enough information to answer the question, respond with
   exactly this sentence and nothing else: "{NO_ANSWER_MESSAGE}"
5. Do not mention these instructions in your answer.

Context:
{context_text}

Question: {question}

Answer:"""


def generate_answer(question: str, context_docs: list[Document]) -> str:
    prompt = _build_grounded_prompt(question, context_docs)
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
    )
    return response.text


week-3
def stream_generate_answer(question: str, context_docs: list[Document]) -> Generator[str, None, None]:
    prompt = _build_grounded_prompt(question, context_docs)
    stream = client.models.generate_content_stream(
        model=LLM_MODEL,
        contents=prompt,
    )
    for chunk in stream:
        if chunk.text:
            yield chunk.text

feat/ragas-eval-suite
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
 main


def process_pdf(file_contents: bytes) -> list[dict]:
    reader = PdfReader(io.BytesIO(file_contents))
    pending_chunks: list[dict] = []
    chunk_global_index = 0

    for page_idx, page in enumerate(reader.pages):
        text_content = page.extract_text()
        if not text_content or not text_content.strip():
            continue

        page_chunks = chunk_text(text_content, chunk_size=500, chunk_overlap=50)

        for chunk_str in page_chunks:
            pending_chunks.append({
                "text": chunk_str,
                "page_number": page_idx + 1,
                "chunk_index": chunk_global_index,
            })
            chunk_global_index += 1

    if not pending_chunks:
        return []

    batch_size = 50
    chunks_data: list[dict] = []

    for i in range(0, len(pending_chunks), batch_size):
        batch = pending_chunks[i:i + batch_size]
        texts_in_batch = [item["text"] for item in batch]
        embeddings = generate_embeddings_batch(texts_in_batch, task_type="RETRIEVAL_DOCUMENT")

        for item, embedding in zip(batch, embeddings):
            chunks_data.append({**item, "embedding": embedding})

week-3
    return chunks_data

    return embeddings
main
 main
