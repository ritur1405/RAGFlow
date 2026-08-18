from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, String, Text, DateTime, func
from .database import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    content = Column(Text, nullable=False)
    # 768 dimensions matches Google Gemini's text-embedding-004 model
    embedding = Column(Vector(768))
    created_at = Column(DateTime(timezone=True), server_default=func.now())