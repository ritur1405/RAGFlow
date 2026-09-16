import re
from typing import Optional
from pydantic import BaseModel, Field, field_validator


# ---- Ingestion schemas ----

class DocumentCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=300)
    content: str = Field(..., min_length=1, max_length=200_000)


class DocumentChunkOut(BaseModel):
    id: int
    title: str
    content: str
    file_name: Optional[str] = None
    chunk_index: Optional[int] = None
    page_number: Optional[int] = None
    relevance_score: Optional[float] = None


class IngestionResponse(BaseModel):
    message: str
    chunks: list[DocumentChunkOut]


# ---- Query schemas ----

# Strips characters with no legitimate place in a natural-language question:
# control chars, null bytes, and common prompt-injection delimiter patterns.
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The user's natural-language question about ingested documents.",
    )
    top_k: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Number of top matching chunks to retrieve.",
    )

    @field_validator("question")
    @classmethod
    def sanitize_question(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Question cannot be empty or whitespace only.")
        v = _CONTROL_CHAR_RE.sub("", v)
        # Collapse excessive internal whitespace (e.g. pasted PDFs, spam padding)
        v = re.sub(r"\s+", " ", v)
        return v


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[DocumentChunkOut]