import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key) if api_key else None


def generate_embedding(text: str):
    if not client:
        raise ValueError("GEMINI_API_KEY is not configured in .env file.")

    response = client.models.embed_content(
        model="text-embedding-004",
        contents=text,
    )

    if hasattr(response, "embedding") and response.embedding:
        return response.embedding.values
    elif hasattr(response, "embeddings") and response.embeddings:
        return response.embeddings[0].values
    else:
        raise ValueError("Unable to extract embedding values from response.")


def generate_answer(query: str, context_documents: list) -> str:
    if not client:
        raise ValueError("GEMINI_API_KEY is not configured in .env file.")

    # Combine context documents into a single prompt string
    context_text = "\n\n".join(
        [f"Title: {doc.title}\nContent: {doc.content}" for doc in context_documents]
    )

    prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided context below.

Context:
{context_text}

Question: {query}
Answer:"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text