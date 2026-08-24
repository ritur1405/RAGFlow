import io
import re

from pypdf import PdfReader


def chunk_text(
    text: str, chunk_size: int = 500, chunk_overlap: int = 50
) -> list[str]:
    """Splits a string into overlapping chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size.")

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


# ---------------------------------------------------------------------------
# Week 2 — PDF extraction, text cleaning, and page-aware chunking
# ---------------------------------------------------------------------------


def extract_text_from_pdf(file_bytes: bytes) -> list[dict]:
    """Extracts text from each page of a PDF.

    Args:
        file_bytes: Raw bytes of the PDF file.

    Returns:
        A list of dicts, one per non-empty page:
        [{"page_number": 1, "text": "..."}, ...]

    Raises:
        ValueError: If the file contains no extractable text or is not a valid PDF.
    """
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception as e:
        raise ValueError(f"Invalid or corrupt PDF file: {e}") from e

    if len(reader.pages) == 0:
        raise ValueError("PDF has no pages.")

    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page_number": i + 1, "text": text})

    if not pages:
        raise ValueError(
            "PDF has pages but no extractable text (may be image-only)."
        )

    return pages


def clean_text(text: str | None) -> str:
    """Cleans raw PDF-extracted text.

    Applies the following transformations in order:
    1. Rejoin hyphenated line breaks  (e.g. "knowl-\\nedge" → "knowledge")
    2. Replace soft newlines with a space (preserves paragraph breaks)
    3. Collapse multiple blank lines into one paragraph separator
    4. Collapse runs of spaces / tabs into a single space
    5. Strip leading and trailing whitespace
    """
    if not text:
        return ""

    # 1. Rejoin hyphenated line breaks
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # 2. Replace single newlines (soft wraps) with spaces
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # 3. Collapse multiple newlines into a paragraph separator
    text = re.sub(r"\n{2,}", "\n\n", text)
    # 4. Collapse runs of spaces / tabs
    text = re.sub(r"[ \t]+", " ", text)
    # 5. Strip
    text = text.strip()

    return text


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[dict]:
    """Splits page texts into overlapping chunks, preserving page metadata.

    Each page is chunked independently so that every chunk has an unambiguous
    ``page_number``.  A global ``chunk_index`` is assigned across all pages.

    Args:
        pages: Output of ``extract_text_from_pdf`` — list of
               ``{"page_number": int, "text": str}`` dicts.
        chunk_size: Maximum character length of each chunk.
        chunk_overlap: Number of overlapping characters between chunks.

    Returns:
        A list of dicts:
        [{"content": str, "page_number": int, "chunk_index": int}, ...]
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size.")

    all_chunks: list[dict] = []
    global_index = 0

    for page in pages:
        text = page["text"]
        page_number = page["page_number"]
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + chunk_size
            chunk_content = text[start:end].strip()
            if chunk_content:
                all_chunks.append(
                    {
                        "content": chunk_content,
                        "page_number": page_number,
                        "chunk_index": global_index,
                    }
                )
                global_index += 1
            start += chunk_size - chunk_overlap

    return all_chunks