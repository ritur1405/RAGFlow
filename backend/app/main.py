from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import Base, engine, get_db
from app.models import Document
from app.services import generate_answer, generate_embedding, search_documents
from app.utils import chunk_text

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


# Request & Response Schemas
class DocumentCreate(BaseModel):
    title: str
    content: str


class DocumentChunkOut(BaseModel):
    id: int
    title: str
    content: str


class IngestionResponse(BaseModel):
    message: str
    chunks: list[DocumentChunkOut]


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[DocumentChunkOut]


@app.get("/")
def read_root():
    return {"message": "RAGFlow Backend API is running!"}


@app.post("/documents/", response_model=IngestionResponse)
def create_document(doc: DocumentCreate, db: Session = Depends(get_db)):
    """Ingests a long document, breaks it into overlapping chunks, generates embeddings, and saves them."""
    if not doc.content.strip():
        raise HTTPException(status_code=400, detail="Content cannot be empty.")

    # Break long content into smaller overlapping chunks
    chunks = chunk_text(doc.content, chunk_size=500, chunk_overlap=50)
    created_docs = []

    for i, chunk in enumerate(chunks):
        embedding = generate_embedding(chunk, task_type="RETRIEVAL_DOCUMENT")
        db_doc = Document(
            title=f"{doc.title} (Chunk {i+1})",
            content=chunk,
            embedding=embedding,
        )
        db.add(db_doc)
        created_docs.append(db_doc)

    db.commit()

    # Refresh instances to capture generated database IDs
    for d in created_docs:
        db.refresh(d)

    return IngestionResponse(
        message=f"Successfully created and stored {len(created_docs)} chunk(s).",
        chunks=[
            DocumentChunkOut(id=d.id, title=d.title, content=d.content)
            for d in created_docs
        ],
    )


@app.get("/documents/", response_model=list[DocumentChunkOut])
def list_documents(db: Session = Depends(get_db)):
    """Retrieves all stored document chunks from the database."""
    docs = db.query(Document).all()
    return [
        DocumentChunkOut(id=d.id, title=d.title, content=d.content)
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
        raise HTTPException(
            status_code=400, detail="Question cannot be empty."
        )

    # 1. Generate embedding for query
    query_vector = generate_embedding(request.question, task_type="RETRIEVAL_QUERY")

    # 2. Search vector database for top matching chunks
    relevant_docs = search_documents(db, query_vector, top_k=3)

    if not relevant_docs:
        return QueryResponse(
            question=request.question,
            answer="No relevant documents found in the database to answer your question.",
            sources=[],
        )

    # 3. Extract relevant content
    context_list = [doc.content for doc in relevant_docs]

    # 4. Generate answer with Gemini
    answer = generate_answer(request.question, context_list)

    return QueryResponse(
        question=request.question,
        answer=answer,
        sources=[
            DocumentChunkOut(id=doc.id, title=doc.title, content=doc.content)
            for doc in relevant_docs
        ],
    )