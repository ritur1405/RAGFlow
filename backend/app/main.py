from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="RAGFlow Backend")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production (e.g. ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants for File Validation
MAX_FILE_SIZE_MB = 10
ALLOWED_MIME_TYPES = ["application/pdf"]


# Global Exception Handler
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
    """Ingests raw text, breaks it into overlapping chunks, generates embeddings, and saves them."""
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
                file_name=getattr(d, "file_name", None),
                chunk_index=d.chunk_index,
                page_number=getattr(d, "page_number", None),
            )
            for d in created_docs
        ],
    )


@app.post("/documents/upload/", response_model=IngestionResponse)
async def upload_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Receives a PDF file, validates format and size, extracts chunks, and saves vectors to PostgreSQL."""

    # 1. Validate File Extension and MIME Type
    if not file.filename.lower().endswith(".pdf") or file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Invalid file format. Only PDF files (.pdf) are supported."
        )

    # 2. Read Binary Content
    contents = await file.read()

    # 3. Validate File Size (Max 10MB)
    file_size_mb = len(contents) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds maximum limit of {MAX_FILE_SIZE_MB}MB."
        )

    # 4. Extract Chunks via PDF Processor
    try:
        parsed_chunks = process_pdf(contents)
    except Exception as e:
        raise HTTPException(
            status_code=422,
            detail=f"Unable to process PDF. File may be corrupted or unreadable. Error: {str(e)}"
        )

    # 5. Handle Scanned / Textless PDFs
    if not parsed_chunks:
        raise HTTPException(
            status_code=400,
            detail="No extractable text found in PDF. Scanned images or image-only PDFs are not supported without OCR."
        )

    created_docs = []

    # 6. Save Chunks, Vector Embeddings, and Metadata in PostgreSQL
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
                file_name=d.file_name,
                chunk_index=d.chunk_index,
                page_number=d.page_number,
            )
            for d in created_docs
        ],
    )


@app.get("/documents/", response_model=list[DocumentChunkOut])
def list_documents(db: Session = Depends(get_db)):
    """Retrieves all stored document chunks from the database."""
    docs = db.query(Document).all()
    return [
        DocumentChunkOut(
            id=d.id,
            title=d.title,
            content=d.content,
            file_name=getattr(d, "file_name", None),
            chunk_index=getattr(d, "chunk_index", None),
            page_number=getattr(d, "page_number", None),
        )
        for d in docs
    ]


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: int, db: Session = Depends(get_db)):
    """Deletes a specific document chunk by ID."""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document chunk not found.")

    db.delete(doc)
    db.commit()
    return {"message": f"Document ID {doc_id} successfully deleted."}


# ---------------------------------------------------------------------------
# Helpers shared by all /query/* endpoints
# ---------------------------------------------------------------------------

def _build_source_out(
    doc: Document,
    relevance_score: float | None = None,
) -> DocumentChunkOut:
    """Builds a DocumentChunkOut from a Document + optional score."""
    return DocumentChunkOut(
        id=doc.id,
        title=doc.title,
        content=doc.content,
        file_name=getattr(doc, "file_name", None),
        chunk_index=getattr(doc, "chunk_index", None),
        page_number=getattr(doc, "page_number", None),
        relevance_score=round(relevance_score, 4) if relevance_score is not None else None,
    )


def _handle_meta_question(
    db: Session, question: str
) -> QueryResponse | None:
    """Returns a QueryResponse for summary-style questions, or None if not applicable."""
    if not is_meta_question(question):
        return None

    answer, relevant_docs = generate_summary_answer(db, question)
    return QueryResponse(
        question=question,
        answer=answer,
        sources=[_build_source_out(doc) for doc in relevant_docs],
    )


def _dense_search_and_answer(
    db: Session, question: str, top_k: int
) -> QueryResponse:
    """Core dense-retrieval query flow: embed → search → threshold → generate."""
    query_vector = generate_embedding(question, task_type="RETRIEVAL_QUERY")
    scored_docs = search_documents(db, query_vector, top_k=top_k)

    # Fallback: no chunks at all, or nothing close enough to be trustworthy.
    if not scored_docs or scored_docs[0][1] > MAX_RELEVANT_DISTANCE:
        return QueryResponse(
            question=question,
            answer=NO_ANSWER_MESSAGE,
            sources=[],
        )

    relevant_docs = [doc for doc, _dist in scored_docs]
    answer = generate_answer(question, relevant_docs)

    return QueryResponse(
        question=question,
        answer=answer,
        sources=[
            _build_source_out(doc, relevance_score=1 - dist)
            for doc, dist in scored_docs
        ],
    )





# ---------------------------------------------------------------------------
# Query endpoints
# ---------------------------------------------------------------------------

@app.post("/query/", response_model=QueryResponse)
def query_rag(request: QueryRequest, db: Session = Depends(get_db)):
    """Default RAG query — uses dense (vector) retrieval.

    - Broad/summary-style questions bypass vector search and use the whole document.
    - Specific questions use top-k pgvector cosine search, but fall back to a fixed
      "cannot find" message if nothing beats MAX_RELEVANT_DISTANCE.
    - request.question is validated/sanitized by QueryRequest (schemas.py).
    """
    meta = _handle_meta_question(db, request.question)
    if meta is not None:
        return meta
    return _dense_search_and_answer(db, request.question, request.top_k)


@app.post("/query/dense", response_model=QueryResponse)
def query_dense(request: QueryRequest, db: Session = Depends(get_db)):
    """Explicit dense (vector) retrieval endpoint for benchmarking."""
    return _dense_search_and_answer(db, request.question, request.top_k)
