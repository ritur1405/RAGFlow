from app.utils import extract_text_from_pdf, clean_text, chunk_pages
from app.services import generate_embeddings_batch

def process_pdf_document(file_bytes: bytes) -> list[dict]:
    """Orchestrates the ingestion pipeline for a PDF file.

    Steps:
        1. Extracts raw text page-by-page.
        2. Cleans text per page.
        3. Chunks pages while preserving page numbers and chunk index.
        4. Generates batch embeddings for all chunks.

    Returns:
        A list of dicts matching Member B's required structure:
        [
            {
                "content": str,
                "page_number": int,
                "chunk_index": int,
                "embedding": list[float]
            },
            ...
        ]
    """
    # 1. Extract text and page numbers
    pages = extract_text_from_pdf(file_bytes)

    # 2. Clean text per page
    cleaned_pages = []
    for page in pages:
        cleaned = clean_text(page["text"])
        if cleaned:
            cleaned_pages.append({
                "page_number": page["page_number"],
                "text": cleaned
            })

    if not cleaned_pages:
        raise ValueError("No text remaining after cleaning extracted PDF pages.")

    # 3. Chunk pages
    chunks = chunk_pages(cleaned_pages, chunk_size=500, chunk_overlap=50)

    # 4. Generate batch embeddings
    contents = [c["content"] for c in chunks]
    embeddings = generate_embeddings_batch(contents, task_type="RETRIEVAL_DOCUMENT", batch_size=100)

    # 5. Verify lengths match before assembling
    if len(embeddings) != len(chunks):
        raise ValueError(f"Embedding count ({len(embeddings)}) does not match chunk count ({len(chunks)}).")

    # 6. Combine and assign to final structure
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb

    return chunks
