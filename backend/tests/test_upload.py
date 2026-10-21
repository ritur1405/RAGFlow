"""Tests for how POST /documents/upload/ reports failures.

A PDF that cannot be parsed is the user's problem (422); an embedding service
that is rate-limited or failing is not, and must not be reported as a corrupt
file. process_pdf is patched, so no PDF parsing or Gemini call happens.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from google.genai import errors as genai_errors

from app import services
from app.database import get_db
from app.main import app


@pytest.fixture
def client():
    app.dependency_overrides[get_db] = lambda: MagicMock()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _upload(client):
    return client.post(
        "/documents/upload/",
        files={"file": ("paper.pdf", b"%PDF-1.5 minimal", "application/pdf")},
    )


def _api_error(cls, code, message):
    return cls(code, {"error": {"code": code, "message": message, "status": "X"}})


def test_rate_limit_is_503_not_a_corrupt_file(client):
    err = _api_error(genai_errors.ClientError, 429, "Quota exceeded for embed_content")
    with patch("app.main.process_pdf", side_effect=err):
        resp = _upload(client)

    assert resp.status_code == 503
    assert "rate limit or quota" in resp.json()["detail"]
    assert "corrupted" not in resp.json()["detail"]


def test_other_upstream_errors_are_502(client):
    err = _api_error(genai_errors.ServerError, 500, "internal")
    with patch("app.main.process_pdf", side_effect=err):
        resp = _upload(client)

    assert resp.status_code == 502
    assert "embedding service failed" in resp.json()["detail"]


def test_unreadable_pdf_is_still_422(client):
    with patch("app.main.process_pdf", side_effect=ValueError("EOF marker not found")):
        resp = _upload(client)

    assert resp.status_code == 422
    assert "corrupted or unreadable" in resp.json()["detail"]


def test_gemini_client_retries_transient_errors_and_rate_limits():
    retry = services.client._api_client._http_options.retry_options
    assert retry.attempts >= 2
    assert 503 in retry.http_status_codes
    # 429s are handled by app.quota so daily quotas are not retried.
    assert 429 not in retry.http_status_codes
    assert services.client.models.generate_content.__wrapped__
    assert services.client.models.embed_content.__wrapped__
