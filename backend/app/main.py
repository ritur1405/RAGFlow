from fastapi import Depends, FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.database import Base, engine, get_db
from app.models import Document
from app.services import generate_answer, generate_embedding, search_documents, process_pdf
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
                chunk_index=d.chunk_index,
                page_number=getattr(d, "page_number", None),
            )
            for d in created_docs
        ],
    )


@app.post("/documents/upload/", response_model=IngestionResponse)
async def upload_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """Receives a PDF file, parses chunks using the processor, and saves vectors to PostgreSQL."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # 1. Read binary bytes from upload
    contents = await file.read()

    # 2. Pass bytes to your teammate's processor function
    try:
        parsed_chunks = process_pdf(contents)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error processing PDF: {str(e)}")

    if not parsed_chunks:
        raise HTTPException(status_code=400, detail="No readable text could be extracted from the PDF.")

    created_docs = []

    # 3. Store extracted chunks and embeddings in PostgreSQL (pgvector)
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


@app.post("/query/", response_model=QueryResponse)
def query_rag(request: QueryRequest, db: Session = Depends(get_db)):
    """Performs vector similarity search over chunks and uses Gemini to answer using context."""
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    query_vector = generate_embedding(request.question, task_type="RETRIEVAL_QUERY")
    relevant_docs = search_documents(db, query_vector, top_k=3)

    if not relevant_docs:
        return QueryResponse(
            question=request.question,
            answer="No relevant documents found in the database to answer your question.",
            sources=[],
        )

    context_list = [doc.content for doc in relevant_docs]
    answer = generate_answer(request.question, context_list)

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
        ]
    )