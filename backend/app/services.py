import io
import os
import time
from typing import Generator

from google import genai
from google.genai import types
from pypdf import PdfReader
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.models import Document, ExperimentRun
from app.utils import chunk_text

# Initialize Gemini Client (reads GEMINI_API_KEY from environment)
client = genai.Client()

EMBEDDING_MODEL = "text-embedding-004"
LLM_MODEL = "gemini-1.5-flash"
EMBEDDING_DIM = 768

MAX_RELEVANT_DISTANCE = 0.8
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


def is_meta_question(question: str) -> bool:
    """Heuristic check for broad/summary-style questions vs. specific-fact lookups."""
    q = question.lower()
    return any(keyword in q for keyword in META_QUESTION_KEYWORDS)


def generate_embedding(text_content: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """Generates 768-dimensional vector embeddings."""
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
    texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT", batch_size: int = 50
) -> list[list[float]]:
    """Generates vector embeddings for a list of strings in batches."""
    if not texts:
        return []

    all_embeddings = []
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
        all_embeddings.extend([e.values for e in response.embeddings])

    return all_embeddings


def search_documents(
    db: Session, query_vector: list[float], top_k: int = 3
) -> list[tuple[Document, float]]:
    """Executes native pgvector similarity search in Supabase PostgreSQL."""
    sql = text("""
        SELECT id, title, content, page_number, chunk_index,
               embedding <-> CAST(:vector AS vector) AS distance
        FROM documents
        WHERE embedding IS NOT NULL
        ORDER BY distance ASC
        LIMIT :top_k
    """)

    vector_str = f"[{','.join(map(str, query_vector))}]"
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
    ]


def _build_grounded_prompt(question: str, context_docs: list[Document]) -> str:
    context_blocks = [doc.content for doc in context_docs if doc.content]
    context_text = "\n\n".join(context_blocks)

    return f"""You are a strictly document-grounded assistant. Follow these rules exactly:

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
    """Synthesizes a strictly grounded RAG response using Gemini based on retrieved docs."""
    prompt = _build_grounded_prompt(question, context_docs)
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
    )
    return response.text


def generate_summary_answer(
    db: Session, question: str, max_chars: int = 12000
) -> tuple[str, list[Document]]:
    """Handles broad/meta questions by feeding Gemini the document in reading order."""
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

Document content (reading order):
{context_text}

Question: {question}

Answer:"""

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
    )
    return response.text, used_docs


def process_pdf(file_contents: bytes) -> list[dict]:
    """Extracts text from raw PDF bytes, chunks it in batch, generates embeddings, and returns dictionaries."""
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

    texts_in_batch = [item["text"] for item in pending_chunks]
    embeddings = generate_embeddings_batch(texts_in_batch, task_type="RETRIEVAL_DOCUMENT")

    chunks_data: list[dict] = []
    for item, embedding in zip(pending_chunks, embeddings):
        chunks_data.append({**item, "embedding": embedding})

    return chunks_data


def search_bm25_placeholder(db: Session, query: str, top_k: int = 5) -> list[tuple[Document, float]]:
    """Placeholder BM25 search until sparse engine integration."""
    stmt = select(Document).limit(top_k)
    results = db.execute(stmt).scalars().all()
    return [(doc, round(1.0 - (i * 0.1), 4)) for i, doc in enumerate(results)]


def run_experiment_comparison(db: Session, query: str, top_k: int = 5, alpha: float = 0.5) -> dict:
    """Executes dense, bm25, and hybrid retrievals side-by-side and logs run metrics."""
    modes = ["dense", "bm25", "hybrid"]
    comparison_data = {}

    query_vector = generate_embedding(query, task_type="RETRIEVAL_QUERY")

    for mode in modes:
        start_time = time.time()

        if mode == "dense":
            raw_results = search_documents(db, query_vector, top_k=top_k)
            formatted_results = [
                {
                    "doc_id": doc.id,
                    "content": doc.content,
                    "file_name": getattr(doc, "title", "document"),
                    "page_number": doc.page_number,
                    "score": round(1.0 - dist, 4),
                }
                for doc, dist in raw_results
            ]
        elif mode == "bm25":
            raw_results = search_bm25_placeholder(db, query, top_k=top_k)
            formatted_results = [
                {
                    "doc_id": doc.id,
                    "content": doc.content,
                    "file_name": getattr(doc, "title", "document"),
                    "page_number": doc.page_number,
                    "score": score,
                }
                for doc, score in raw_results
            ]
        else:  # Hybrid retrieval strategy
            dense_res = search_documents(db, query_vector, top_k=top_k)
            formatted_results = [
                {
                    "doc_id": doc.id,
                    "content": doc.content,
                    "file_name": getattr(doc, "title", "document"),
                    "page_number": doc.page_number,
                    "score": round(((1.0 - dist) * alpha) + (0.5 * (1.0 - alpha)), 4),
                }
                for doc, dist in dense_res
            ]

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        run_log = ExperimentRun(
            query=query,
            mode=mode,
            retrieved_docs=formatted_results,
            execution_time_ms=elapsed_ms,
        )
        db.add(run_log)

        comparison_data[mode] = {
            "mode": mode,
            "execution_time_ms": elapsed_ms,
            "results": formatted_results,
        }

    db.commit()
    return comparison_data