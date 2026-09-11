from datetime import datetime
from sqlalchemy import Column, Integer, String, Text, Float, DateTime, JSON
from pgvector.sqlalchemy import Vector
from app.database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, nullable=True)     # Stores original filename (e.g. sample.pdf)
    title = Column(String, nullable=False)
    content = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=True)  # Page from PDF
    chunk_index = Column(Integer, nullable=True)  # Position index of chunk
    embedding = Column(Vector(768))                # pgvector embedding


# ---- Week 4: Experimentation & Configuration Models ----

class ExperimentConfig(Base):
    __tablename__ = "experiment_configs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False, index=True) # e.g., "default_hybrid"
    retrieval_mode = Column(String, nullable=False)                # "dense", "bm25", "hybrid"
    top_k = Column(Integer, default=5)
    alpha = Column(Float, default=0.5)                              # Weight between Dense (1.0) and BM25 (0.0)
    created_at = Column(DateTime, default=datetime.utcnow)


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    id = Column(Integer, primary_key=True, index=True)
    query = Column(String, nullable=False)
    mode = Column(String, nullable=False)
    retrieved_docs = Column(JSON, nullable=False)                  # Stores list of doc IDs, text snippets, and scores
    execution_time_ms = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)