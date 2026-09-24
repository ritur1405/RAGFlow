import json
from typing import Generator

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.database import Base, engine, get_db
from app.models import Document
from app.routers import eval as eval_router
from app.schemas import QueryRequest, QueryResponse
from app.services import (
    NO_ANSWER_MESSAGE,
    generate_answer,
    generate_embedding,
    generate_summary_answer,
    is_meta_question,
    process_pdf,
    search_documents,
)

# Initialize FastAPI application
app = FastAPI(title="RAGOps API")

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
# Prefix is omitted here because '/api/eval' is already defined inside app/routers/eval.py
app.include_router(eval_router.router, tags=["evaluation"])

# Create database tables (documents + evaluation_*)
Base.metadata.create_all(bind=engine)


@app.get("/")
def read_root():
    return {"status": "ok", "message": "RAGOps API is running"}

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

    if not results:
        return QueryResponse(answer=NO_ANSWER_MESSAGE, retrieved_docs=[])

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Uploads a PDF, chunks text, generates embeddings, and saves to database."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded PDF file is empty.")

    try:
        chunks_data = process_pdf(contents)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to process PDF file: {str(e)}"
        )

    if not chunks_data:
        raise HTTPException(
            status_code=400, detail="Could not extract text from the provided PDF."
        )

    for chunk in chunks_data:
        db.add(
            Document(
                title=file.filename,
                file_name=file.filename,
                content=chunk["text"],
                page_number=chunk["page_number"],
                chunk_index=chunk["chunk_index"],
                embedding=chunk["embedding"],
            )
        )

    db.commit()
    return {
        "message": f"Successfully processed {len(chunks_data)} chunks from {file.filename}."
    }


@app.post("/query/", response_model=QueryResponse)
def query_rag(payload: QueryRequest, db: Session = Depends(get_db)):
    """Standard (non-streaming) grounded RAG endpoint."""
    question = payload.question.strip() if payload.question else ""
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if is_meta_question(question):
        answer, docs = generate_summary_answer(db, question)
        return QueryResponse(
            question=question,
            answer=answer,
            retrieved_docs=docs,
            sources=docs,
        )

    query_vector = generate_embedding(question, task_type="RETRIEVAL_QUERY")
    results = search_documents(db, query_vector, top_k=3)

    if not results:
        return QueryResponse(
            question=question,
            answer=NO_ANSWER_MESSAGE,
            retrieved_docs=[],
            sources=[],
        )

    docs = [doc for doc, _dist in results]
    answer = generate_answer(question, docs)
    return QueryResponse(
        question=question,
        answer=answer,
        sources=[
            DocumentChunkOut(
                id=doc.id,
                title=doc.title,
                content=doc.content,
                chunk_index=getattr(doc, "chunk_index", None),
                page_number=getattr(doc, "page_number", None),
            )
            for doc in relevant_docs
        ],
    )
