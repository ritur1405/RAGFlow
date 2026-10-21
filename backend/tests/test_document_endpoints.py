"""Tests for the document-library endpoints in app/main.py, on SQLite."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app
from app.models import Document


@pytest.fixture
def db_session(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'docs.db'}")
    Base.metadata.create_all(bind=engine, tables=[Document.__table__])
    session = sessionmaker(bind=engine, autocommit=False, autoflush=False)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture
def client(db_session):
    app.dependency_overrides[get_db] = lambda: db_session
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _add(db, file_name, page, title=None):
    db.add(Document(
        file_name=file_name, title=title or f"{file_name} p{page}", content="text",
        page_number=page, chunk_index=0,
    ))
    db.commit()


@pytest.fixture
def library(db_session):
    for page in (1, 1, 2, 3):
        _add(db_session, "paper.pdf", page)
    _add(db_session, "notes.pdf", 1)
    _add(db_session, None, None, title="Pasted text")
    return db_session


def test_lists_one_row_per_file_newest_first(client, library):
    files = client.get("/documents/files").json()

    assert [f["file_name"] for f in files] == ["Pasted text", "notes.pdf", "paper.pdf"]
    paper = files[-1]
    assert (paper["chunks"], paper["pages"], paper["is_file"]) == (4, 3, True)
    assert files[0]["is_file"] is False


def test_empty_library(client):
    assert client.get("/documents/files").json() == []


def test_delete_by_file_removes_only_that_file(client, library):
    resp = client.delete("/documents/by-file/paper.pdf")

    assert resp.status_code == 200
    assert resp.json()["deleted"] == 4
    assert [f["file_name"] for f in client.get("/documents/files").json()] == [
        "Pasted text", "notes.pdf",
    ]


def test_delete_unknown_file_is_404(client, library):
    assert client.delete("/documents/by-file/missing.pdf").status_code == 404


def test_chunk_listing_includes_file_name(client, library):
    chunks = client.get("/documents/").json()
    assert {c["file_name"] for c in chunks} == {"paper.pdf", "notes.pdf", None}


def test_upload_response_reports_each_chunk_file_name(client, db_session):
    from unittest.mock import patch

    chunks = [
        {"text": "first", "content": "first", "page_number": 1, "chunk_index": 0, "embedding": [0.1] * 768},
        {"text": "second", "content": "second", "page_number": 2, "chunk_index": 1, "embedding": [0.2] * 768},
    ]
    with patch("app.main.process_pdf", return_value=chunks):
        resp = client.post(
            "/documents/upload/",
            files={"file": ("paper.pdf", b"%PDF-1.5", "application/pdf")},
        )

    assert resp.status_code == 200
    body = resp.json()["chunks"]
    assert [(c["file_name"], c["page_number"]) for c in body] == [("paper.pdf", 1), ("paper.pdf", 2)]
    # And the stored rows agree with what the response reported.
    assert {d.file_name for d in db_session.query(Document).all()} == {"paper.pdf"}
