week-3
import json

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
main
from sqlalchemy.orm import Session

from app.database import Base, engine, get_db
from app.models import Document
from app.services import (
    generate_answer,
    generate_embedding,
    search_documents,
    process_pdf,
    is_meta_question,
    generate_summary_answer,
week-3
    stream_generate_answer,

main
    MAX_RELEVANT_DISTANCE,
    NO_ANSWER_MESSAGE,
)
from app.utils import chunk_text
from app.schemas import (
    DocumentCreate,
    DocumentChunkOut,
    IngestionResponse,
    QueryRequest,
    QueryResponse,
)

Base.metadata.create_all(bind=engine)

app = FastAPI(title="RAGFlow Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

week-3
# Constants for File Validation
 main
MAX_FILE_SIZE_MB = 10
ALLOWED_MIME_TYPES = ["application/pdf"]


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred in the pipeline.",
            "details": str(exc),
        },
    )


@app.get("/")
def read_root():
    return {"message": "RAGFlow Backend API is running!"}


@app.post("/documents/", response_model=IngestionResponse)
def create_document(doc: DocumentCreate, db: Session = Depends(get_db)):
week-3

    """Ingests raw text, breaks it into overlapping chunks, generates embeddings, and saves them."""
 main
    if not doc.content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty.")

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
week-3
                file_name=getattr(d, "file_name", None),

 main
                chunk_index=d.chunk_index,
                page_number=getattr(d, "page_number", None),
            )
            for d in created_docs
        ],
    )


@app.post("/documents/upload/", response_model=IngestionResponse)
async def upload_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
week-3

    """Receives a PDF file, validates format and size, extracts chunks, and saves vectors to PostgreSQL."""

    # 1. Validate File Extension and MIME Type
main
    if not file.filename.lower().endswith(".pdf") or file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only PDF files (.pdf) are supported."
        )

 week-3
    contents = await file.read()


    # 2. Read Binary Content
    contents = await file.read()

    # 3. Validate File Size (Max 10MB)
 main
    file_size_mb = len(contents) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds maximum limit of {MAX_FILE_SIZE_MB}MB."
        )

week-3
    # 4. Extract Chunks via PDF Processor
 main
    try:
        parsed_chunks = process_pdf(contents)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Unable to process PDF. File may be corrupted or unreadable. Error: {str(e)}"
        )

week-3

    # 5. Handle Scanned / Textless PDFs
main
    if not parsed_chunks:
        raise HTTPException(
            status_code=400,
            detail="No extractable text found in PDF. Scanned images or image-only PDFs are not supported without OCR."
        )

    created_docs = []

 week-3

    # 6. Save Chunks, Vector Embeddings, and Metadata in PostgreSQL
 main
    for item in parsed_chunks:
        db_doc = Document(
            file_name=file.filename,
            title=f"{file.filename} (Page {item.get('page_number', 1)}, Chunk {item.get('chunk_index', 0)})",
            content=item.get("text", ""),
            page_number=item.get("page_number"),
            chunk_index=item.get("chunk_index"),
            embedding=item.get("embedding"),
        )
        db.add(db_doc)
        created_docs.append(db_doc)

    db.commit()

    for d in created_docs:
        db.refresh(d)

    return IngestionResponse(
        message=f"Successfully processed '{file.filename}' and stored {len(created_docs)} chunk(s).",
        chunks=[
            DocumentChunkOut(
                id=d.id,
                title=d.title,
                content=d.content,
 week-3
                file_name=d.file_name,

 main
                chunk_index=d.chunk_index,
                page_number=d.page_number,
            )
            for d in created_docs
        ],
    )


@app.get("/documents/", response_model=list[DocumentChunkOut])
def list_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).all()
    return [
        DocumentChunkOut(
            id=d.id,
            title=d.title,
            content=d.content,
 week-3
            file_name=getattr(d, "file_name", None),

 main
            chunk_index=getattr(d, "chunk_index", None),
            page_number=getattr(d, "page_number", None),
        )
        for d in docs
    ]


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document chunk not found.")

    db.delete(doc)
    db.commit()
    return {"message": f"Document ID {doc_id} successfully deleted."}


 week-3
def _resolve_context(request: QueryRequest, db: Session):
    if is_meta_question(request.question):
        answer, relevant_docs = generate_summary_answer(db, request.question)
        return relevant_docs, answer

@app.post("/query/", response_model=QueryResponse)
def query_rag(request: QueryRequest, db: Session = Depends(get_db)):
    """Performs vector similarity search over chunks and uses Gemini to answer using context.

    - Broad/summary-style questions bypass vector search and use the whole document.
    - Specific questions use top-k pgvector search, but fall back to a fixed
      "cannot find" message if nothing beats MAX_RELEVANT_DISTANCE, rather than
      forcing Gemini to answer from irrelevant chunks.
    - request.question is already validated/sanitized by QueryRequest (schemas.py):
      length-bounded and stripped of control characters.
    """
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
                )
                for doc in relevant_docs
            ],
        )
 main

    query_vector = generate_embedding(request.question, task_type="RETRIEVAL_QUERY")
    scored_docs = search_documents(db, query_vector, top_k=3)

 week-3
    if not scored_docs or scored_docs[0][1] > MAX_RELEVANT_DISTANCE:
        return [], NO_ANSWER_MESSAGE

    relevant_docs = [doc for doc, distance in scored_docs]
    return relevant_docs, None


def _to_chunk_out(doc: Document) -> DocumentChunkOut:
    return DocumentChunkOut(
        id=doc.id,
        title=doc.title,
        content=doc.content,
        file_name=getattr(doc, "file_name", None),
        chunk_index=getattr(doc, "chunk_index", None),
        page_number=getattr(doc, "page_number", None),
    )


@app.post("/query/", response_model=QueryResponse)
def query_rag(request: QueryRequest, db: Session = Depends(get_db)):
    relevant_docs, fallback_answer = _resolve_context(request, db)

    if fallback_answer is not None:
        return QueryResponse(
            question=request.question,
            answer=fallback_answer,
            sources=[_to_chunk_out(doc) for doc in relevant_docs],
        )

    answer = generate_answer(request.question, relevant_docs)

    # Fallback: no chunks at all, or nothing close enough to be trustworthy.
    if not scored_docs or scored_docs[0][1] > MAX_RELEVANT_DISTANCE:
        return QueryResponse(
            question=request.question,
            answer=NO_ANSWER_MESSAGE,
            sources=[],
        )

    relevant_docs = [doc for doc, distance in scored_docs]
    context_list = [doc.content for doc in relevant_docs]
    answer = generate_answer(request.question, context_list)
 main

    return QueryResponse(
        question=request.question,
        answer=answer,
 week-3
        sources=[_to_chunk_out(doc) for doc in relevant_docs],
    )


@app.post("/query/stream/")
def query_rag_stream(request: QueryRequest, db: Session = Depends(get_db)):
    relevant_docs, fallback_answer = _resolve_context(request, db)

    def event_stream():
        if fallback_answer is not None:
            safe_text = fallback_answer.replace("\n", "\ndata: ")
            yield f"data: {safe_text}\n\n"
        else:
            for token in stream_generate_answer(request.question, relevant_docs):
                safe_token = token.replace("\n", "\ndata: ")
                yield f"data: {safe_token}\n\n"

        sources_payload = json.dumps([
            _to_chunk_out(doc).model_dump() for doc in relevant_docs
        ])
        yield f"event: sources\ndata: {sources_payload}\n\n"
        yield "event: done\ndata: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

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
 main
