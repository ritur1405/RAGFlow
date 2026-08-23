import os
from google import genai
from google.genai import types
from sqlalchemy import text
from sqlalchemy.orm import Session
from app.models import Document

# Initialize Gemini Client using environment variable key
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768  # must match pgvector column: Vector(768) in models.py


def generate_embedding(text_content: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """Generates 768-dimensional vector embeddings using gemini-embedding-001.

    task_type should be "RETRIEVAL_DOCUMENT" when embedding chunks being stored,
    and "RETRIEVAL_QUERY" when embedding an incoming search question. Gemini
    embeddings are asymmetric, so matching this to usage improves retrieval quality.
    """
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text_content,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBEDDING_DIM,
        ),
    )
    # response.embeddings is a list[ContentEmbedding]; .values is the float vector
    return response.embeddings[0].values


def search_documents(
    db: Session, query_vector: list[float], top_k: int = 3
) -> list[Document]:
    """Executes pgvector similarity search to fetch matching chunks."""
    vector_str = f"[{','.join(map(str, query_vector))}]"

    sql = text("""
        SELECT id, title, content
        FROM documents
        ORDER BY embedding <-> :vector ASC
        LIMIT :top_k
    """)

    results = db.execute(sql, {"vector": vector_str, "top_k": top_k}).fetchall()

    return [
        Document(id=row.id, title=row.title, content=row.content)
        for row in results
    ]


def generate_answer(question: str, context_chunks: list[str]) -> str:
    """Uses Gemini to synthesize an answer based on retrieved document chunks."""
    context_text = "\n\n".join(context_chunks)
    prompt = f"""You are a helpful assistant. Use the provided context to answer the question.
If the answer cannot be found in the context, state that you do not know based on the provided information.

Context:
{context_text}

Question: {question}
Answer:"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )
    return response.text