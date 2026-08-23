def chunk_text(
    text: str, chunk_size: int = 500, chunk_overlap: int = 50
) -> list[str]:
    """Splits a string into overlapping chunks."""
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        # Move forward by chunk_size minus overlap
        start += chunk_size - chunk_overlap

    return chunks