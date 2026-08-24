# Member A → Member B: PDF Ingestion Pipeline Handoff

## What's Ready

The PDF ingestion pipeline is complete and tested on the `feature/week-2-pdf-ingestion` branch.
Member B can now build the upload endpoint and database storage layer on top of it.

## Entry Point

```python
from app.ingestion import process_pdf_document

chunks = process_pdf_document(file_bytes)  # file_bytes: bytes from UploadFile.read()
```

### Input

| Parameter | Type | Description |
|---|---|---|
| `file_bytes` | `bytes` | Raw PDF file content (e.g. `await upload_file.read()`) |

### Output

Returns `list[dict]` — one dict per chunk:

```python
[
    {
        "content": "The extracted and cleaned text of this chunk...",
        "page_number": 1,       # Original PDF page (1-indexed)
        "chunk_index": 0,       # Global index across the entire PDF
        "embedding": [0.012, -0.034, ...]  # 768-dimensional float vector
    },
    {
        "content": "Next chunk of text...",
        "page_number": 1,
        "chunk_index": 1,
        "embedding": [0.056, 0.078, ...]
    },
    ...
]
```

### Errors Raised

| Error | When |
|---|---|
| `ValueError("Invalid or corrupt PDF file: ...")` | Bytes are not a valid PDF |
| `ValueError("PDF has no pages.")` | PDF has zero pages |
| `ValueError("PDF has pages but no extractable text (may be image-only).")` | All pages are blank or image-only |
| `ValueError("No text remaining after cleaning extracted PDF pages.")` | Cleaning removed all content |
| `ValueError("Embedding count (...) does not match chunk count (...).")` | API returned wrong number of embeddings |

All errors are `ValueError` — Member B can catch this single type in the upload endpoint.

## Pipeline Steps (Internal)

1. **Extract** — `pypdf.PdfReader` reads raw bytes, returns text per page
2. **Clean** — Regex-based: rejoins hyphenated words, normalizes whitespace, preserves paragraphs
3. **Chunk** — 500-char chunks with 50-char overlap, chunked per-page for unambiguous `page_number`
4. **Embed** — Batched Gemini API calls (100 chunks per request) using `gemini-embedding-001` (768-dim)

## Files Member B Should NOT Modify

| File | Owner |
|---|---|
| `backend/app/utils.py` | Member A |
| `backend/app/ingestion.py` | Member A |
| `backend/app/services.py` (batch embedding function) | Member A |
| `backend/tests/test_utils.py` | Member A |
| `backend/tests/test_services.py` | Member A |
| `backend/tests/test_ingestion.py` | Member A |

## What Member B Needs To Do

1. Add `python-multipart` to `requirements.txt` (for `UploadFile`)
2. Create an `UploadFile` endpoint in `main.py` that calls `process_pdf_document(await file.read())`
3. Iterate over the returned chunks and save each to the database (updating `models.py` as needed for `page_number`, `chunk_index`)
4. Return a response with the number of chunks created

## Test Configuration

Tests require no external services. The `conftest.py` sets dummy `DATABASE_URL` and `GEMINI_API_KEY` automatically. Run:

```bash
.venv/Scripts/python -m pytest backend/tests/ -v
```
