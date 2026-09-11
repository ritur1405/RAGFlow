import io
import os
import time
from typing import Generator

from google import genai
from google.genai import types
from pypdf import PdfReader
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Document, ExperimentRun
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


def generate_embedding(text_content: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
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
    ]
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
    prompt = _build_grounded_prompt(question, context_docs)
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
    )
    return response.text


def stream_generate_answer(question: str, context_docs: list[Document]) -> Generator[str, None, None]:
    prompt = _build_grounded_prompt(question, context_docs)
    stream = client.models.generate_content_stream(
        model=LLM_MODEL,
        contents=prompt,
    )
    for chunk in stream:
        if chunk.text:
            yield chunk.text


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

    return chunks_data


# ---- Week 4: Retrieval Experiment Functions ----

def search_bm25_placeholder(db: Session, query: str, top_k: int = 5) -> list[tuple[Document, float]]:
    """Placeholder BM25 search until Member A integrates the sparse BM25 engine."""
    stmt = select(Document).limit(top_k)
    results = db.execute(stmt).scalars().all()
    return [(doc, round(1.0 - (i * 0.1), 4)) for i, doc in enumerate(results)]


def run_experiment_comparison(db: Session, query: str, top_k: int = 5, alpha: float = 0.5) -> dict:
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
                    "file_name": doc.file_name,
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
                    "file_name": doc.file_name,
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
                    "file_name": doc.file_name,
                    "page_number": doc.page_number,
                    "score": round(((1.0 - dist) * alpha) + (0.5 * (1.0 - alpha)), 4),
                }
                for doc, dist in dense_res
            ]

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        # Log run execution
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