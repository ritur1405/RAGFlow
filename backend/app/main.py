from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.database import Base, engine, get_db
from app.models import Document
from app.routers import eval as eval_router
from app.services import (
    generate_embedding,
    process_pdf,
    is_meta_question,
    generate_summary_answer,
)
from app.config import NO_ANSWER_MESSAGE
from app.rag_pipeline import query_rag
from app.utils import chunk_text
from app.schemas import (
    DocumentCreate,
    DocumentChunkOut,
    IngestionResponse,
    QueryRequest,
    QueryResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Creates tables at startup rather than at import time.

    Importing this module -- which the test suite and any tooling does -- must
    not require a reachable database. Table creation belongs to starting the
    app, not to loading it.
    """
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="RAGFlow Backend", lifespan=lifespan)

# Week 6: generation-evaluation endpoints under /api/eval.
app.include_router(eval_router.router)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"status": "ok", "message": "RAGFlow API is running"}

    chunks = chunk_text(doc.content, chunk_size=500, chunk_overlap=50)
    created_docs = []

    for i, chunk in enumerate(chunks):
        embedding = generate_embedding(chunk, task_type="RETRIEVAL_DOCUMENT")
        db_doc = Document(
            title=f"{doc.title} (Chunk {i+1})",
            content=chunk,
            embedding=embedding,
            chunk_index=i,
        )
        db.add(db_doc)
        created_docs.append(db_doc)

    db.commit()

    for d in created_docs:
        db.refresh(d)

    return IngestionResponse(
        message=f"Successfully created and stored {len(created_docs)} chunk(s).",
        chunks=[
            DocumentChunkOut(
                id=d.id,
                title=d.title,
                content=d.content,
                chunk_index=d.chunk_index,
                page_number=getattr(d, "page_number", None),
            )
            for d in created_docs
        ],
    )


# /upload/ is an alias kept for the Week 6 frontend, which was written against
# that shorter path. /documents/upload/ remains the canonical route; both are
# the same handler, so neither contract can drift from the other.
@app.post("/documents/upload/", response_model=IngestionResponse)
@app.post("/upload/", response_model=IngestionResponse, include_in_schema=False)
async def upload_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Uploads a PDF, chunks text, generates embeddings, and saves to database."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = await file.read()
    chunks_data = process_pdf(contents)

    if not chunks_data:
        raise HTTPException(
            status_code=400, detail="Could not extract text from the provided PDF."
        )

    # Save chunks into DB
    from app.models import Document

    for chunk in chunks_data:
        doc = Document(
            title=file.filename,
            content=chunk["text"],
            page_number=chunk["page_number"],
            chunk_index=chunk["chunk_index"],
            embedding=chunk["embedding"],
        )
        db.add(doc)

    db.commit()
    return {
        "message": f"Successfully processed {len(chunks_data)} chunks from {file.filename}."
    }


@app.post("/query/", response_model=QueryResponse)
def query_rag(payload: QueryRequest, db: Session = Depends(get_db)):
    """Standard (non-streaming) grounded RAG endpoint."""
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if is_meta_question(question):
        answer, docs = generate_summary_answer(db, question)
        return QueryResponse(answer=answer, retrieved_docs=docs)

    query_vector = generate_embedding(question, task_type="RETRIEVAL_QUERY")
    results = search_documents(db, query_vector, top_k=3)

    if not results:
        return QueryResponse(answer=NO_ANSWER_MESSAGE, retrieved_docs=[])

    docs = [doc for doc, dist in results]
    answer = generate_answer(question, docs)
    return QueryResponse(answer=answer, retrieved_docs=docs)


@app.post("/query/stream/")
def query_rag_stream(payload: QueryRequest, db: Session = Depends(get_db)):
    """SSE streaming endpoint for RAG response generation."""
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

@app.post("/query/", response_model=QueryResponse)
def query_endpoint(request: QueryRequest, db: Session = Depends(get_db)):
    """Performs RAG query: retrieval → context → LLM → answer with citations.

    - Broad/summary-style questions bypass vector search and use the whole document.
    - Specific questions use the dense RAG pipeline (embedding → pgvector → LLM).
    - request.question is already validated/sanitized by QueryRequest (schemas.py):
      length-bounded and stripped of control characters.
    """
    # Meta/summary questions still use the full-document path.
    if is_meta_question(request.question):
        answer, relevant_docs = generate_summary_answer(db, request.question)
        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=[
                DocumentChunkOut(
                    id=doc.id,
                    title=doc.title,
                    content=doc.content,
                    chunk_index=getattr(doc, "chunk_index", None),
                    page_number=getattr(doc, "page_number", None),
                    file_name=getattr(doc, "file_name", None),
                )
                for doc in relevant_docs
            ],
        )

    # Dense RAG pipeline: query → embedding → retrieval → context → LLM → answer.
    result = query_rag(db, request.question)

    return QueryResponse(
        question=request.question,
        answer=result.answer,
        sources=[
            DocumentChunkOut(
                id=src.chunk_id,
                title=src.document_name or "Unknown",
                content=src.text,
                chunk_index=src.chunk_index,
                page_number=src.page_number,
                file_name=src.document_name,
                score=src.score,
            )
            for src in result.sources
        ],
    )
