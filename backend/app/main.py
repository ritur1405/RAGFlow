import json
from typing import Generator

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.orm import Session

from app.database import Base, engine, get_db
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

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="RAGFlow Backend")

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


@app.post("/upload/")
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

    query_vector = generate_embedding(question, task_type="RETRIEVAL_QUERY")
    results = search_documents(db, query_vector, top_k=3)

    docs = [doc for doc, dist in results] if results else []

    def event_generator() -> Generator[str, None, None]:
        if not docs:
            yield f"data: {json.dumps({'type': 'token', 'content': NO_ANSWER_MESSAGE})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            return

        full_answer = generate_answer(question, docs)
        for chunk in full_answer.split(" "):
            yield f"data: {json.dumps({'type': 'token', 'content': chunk + ' '})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")