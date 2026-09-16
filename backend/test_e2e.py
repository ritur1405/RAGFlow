"""
test_e2e.py — RAGFlow end-to-end integration check.

Benchmarks the RAG pipeline on the full request cycle:
  1. Upload a PDF          -> POST /documents/upload/
  2. Confirm ingestion      -> GET  /documents/
  3. Ask a specific question -> POST /query/       (default dense retrieval)
  4. Ask via /query/dense    -> POST /query/dense   (explicit dense retrieval)
  5. Ask a broad question    -> POST /query/        (meta-question / summary path)

This uses FastAPI's TestClient, so it runs the app in-process — no need to
have `uvicorn` running separately. It DOES make real calls to Gemini and your
real Supabase database, so it costs API quota and takes a few seconds; this
is an integration check, not a fast unit test.

Usage:
    cd backend
    python test_e2e.py --pdf path/to/your_file.pdf --question "your question here"

If no --pdf/--question are given, sensible defaults are used (see below) —
point --pdf at any real PDF you have locally to exercise the pipeline against
fresh content.
"""

import argparse
import os
import sys

from fastapi.testclient import TestClient

# Ensure `app` package resolves the same way it does when running uvicorn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.main import app  # noqa: E402

client = TestClient(app)


def upload_pdf(pdf_path: str) -> dict:
    print(f"\n[1/5] Uploading PDF: {pdf_path}")
    if not os.path.exists(pdf_path):
        print(f"ERROR: file not found at {pdf_path}")
        sys.exit(1)

    with open(pdf_path, "rb") as f:
        response = client.post(
            "/documents/upload/",
            files={"file": (os.path.basename(pdf_path), f, "application/pdf")},
        )

    if response.status_code != 200:
        print(f"FAILED — status {response.status_code}: {response.text}")
        sys.exit(1)

    data = response.json()
    print(f"OK — {data['message']}")
    return data


def confirm_ingestion(expected_min_chunks: int = 1) -> list[dict]:
    print("\n[2/5] Confirming ingested documents via GET /documents/")
    response = client.get("/documents/")

    if response.status_code != 200:
        print(f"FAILED — status {response.status_code}: {response.text}")
        sys.exit(1)

    docs = response.json()
    print(f"OK — {len(docs)} chunk(s) currently stored")

    if len(docs) < expected_min_chunks:
        print("WARNING: fewer chunks than expected — ingestion may be incomplete.")

    # Verify new metadata fields exist in the response
    if docs:
        sample = docs[0]
        has_file_name = "file_name" in sample
        print(f"  file_name field present: {has_file_name}")

    return docs


def _print_sources(data: dict) -> None:
    """Pretty-prints the sources from a query response."""
    if data["sources"]:
        print("Sources cited:")
        for src in data["sources"]:
            score_str = ""
            if src.get("relevance_score") is not None:
                score_str = f", score={src['relevance_score']:.4f}"
            print(
                f"  - {src['title']} "
                f"(file={src.get('file_name')}, page={src.get('page_number')}, "
                f"chunk={src.get('chunk_index')}{score_str})"
            )
    else:
        print("Sources cited: none")


def ask_question(question: str, label: str, endpoint: str = "/query/") -> dict:
    print(f"\n{label} [{endpoint}] Asking: \"{question}\"")
    response = client.post(endpoint, json={"question": question})

    if response.status_code != 200:
        print(f"FAILED — status {response.status_code}: {response.text}")
        sys.exit(1)

    data = response.json()
    print(f"Answer: {data['answer']}")
    _print_sources(data)
    return data


def main():
    parser = argparse.ArgumentParser(description="RAGFlow end-to-end integration check")
    parser.add_argument(
        "--pdf",
        default="TechBlog_Draft.pdf",
        help="Path to a PDF file to upload and ingest (default: TechBlog_Draft.pdf in this folder)",
    )
    parser.add_argument(
        "--question",
        default="How does the search engine combine vector search and BM25?",
        help="A specific, fact-based question the uploaded PDF should be able to answer",
    )
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY is not set in the environment.")
        sys.exit(1)

    print("=" * 70)
    print("RAGFlow End-to-End Integration Check (Week 3)")
    print("=" * 70)

    upload_pdf(args.pdf)
    confirm_ingestion()

    # Default query (dense retrieval)
    ask_question(args.question, "[3/5]", "/query/")

    # Explicit dense retrieval endpoint
    ask_question(args.question, "[4/5]", "/query/dense")

    # Broad question -> exercises the meta-question / full-document summary path
    ask_question("What is the main topic discussed in this document?", "[5/5]", "/query/")

    print("\n" + "=" * 70)
    print("End-to-end check complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()