from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Document
from app.schemas import DocumentCreate, QueryRequest
from app.services import generate_embedding, generate_answer

app = FastAPI()


@app.post("/documents/")
def create_document(doc: DocumentCreate, db: Session = Depends(get_db)):
    embedding = generate_embedding(doc.content)
    db_doc = Document(title=doc.title, content=doc.content, embedding=embedding)
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)
    return db_doc


@app.get("/documents/")
def get_documents(db: Session = Depends(get_db)):
    return db.query(Document).all()


@app.post("/query/")
def query_documents(query_req: QueryRequest, db: Session = Depends(get_db)):
    query_embedding = generate_embedding(query_req.query)

    # Perform vector similarity search
    results = (
        db.query(Document)
        .order_by(Document.embedding.l2_distance(query_embedding))
        .limit(query_req.top_k)
        .all()
    )

    # Generate answer from retrieved context
    answer = generate_answer(query_req.query, results)

    return {
        "query": query_req.query,
        "answer": answer,
        "source_documents": results,
    }