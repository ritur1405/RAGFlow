"""Shared test fixtures.

Sets dummy environment variables so that module-level code in
services.py and database.py does not crash during test collection.
These are set before any app module is imported.
"""

import os

# Must be set before any app imports trigger module-level initialization
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("GEMINI_API_KEY", "test-dummy-key")

import pytest


@pytest.fixture(autouse=True)
def _no_reranker_warm_up(monkeypatch):
    """Entering the app's lifespan (any `with TestClient(app)`) pre-loads the
    cross-encoder model. Tests stub it out so they never download or load a
    model; test_lifespan_warms_up_the_reranker checks the call itself."""
    import app.main

    monkeypatch.setattr(app.main, "warm_up_reranker", lambda: None)
