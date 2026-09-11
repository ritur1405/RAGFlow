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
