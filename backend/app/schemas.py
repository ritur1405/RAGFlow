from pydantic import BaseModel
from typing import Optional
from datetime import datetime

# Request model when uploading/adding a document
class DocumentCreate(BaseModel):
    title: str
    content: str

# Response model when returning a document from the database
class DocumentResponse(BaseModel):
    id: int
    title: str
    content: str
    created_at: datetime

    class Config:
        from_attributes = True

# Request model for vector search / querying
class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3