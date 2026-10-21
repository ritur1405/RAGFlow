"""Tests for the retrieval-experiment API in app/routers/experiments.py.

The retrievers are patched at the router boundary (run_retriever), so these
tests cover request handling, persistence and failure reporting rather than
retrieval quality, which tests/test_rag_pipelines.py covers. A file-backed
SQLite database provides real persistence for the two experiment tables.
"""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.main import app
from app.models import Document, ExperimentConfig, ExperimentRun


@pytest.fixture
def db_session(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'experiments.db'}")
    Base.metadata.create_all(
        bind=engine, tables=[ExperimentConfig.__table__, ExperimentRun.__table__]
    )
    Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = Session()
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


def _doc(doc_id: int, page: int = 1) -> Document:
    return Document(
        id=doc_id,
        file_name="paper.pdf",
        title="paper.pdf",
        content=f"chunk {doc_id}",
        page_number=page,
        chunk_index=doc_id,
    )


# Distinguishable rankings per strategy, so a mix-up between them is visible.
_RANKINGS = {
    "dense": ([(_doc(1), 0.21), (_doc(2), 0.35)], 120.0),
    "bm25": ([(_doc(3, page=4), 7.5), (_doc(1), 2.0)], 15.0),
    "hybrid": ([(_doc(1), 0.0325), (_doc(3, page=4), 0.0161)], 140.0),
}


@pytest.fixture
def patched_retrievers():
    async def fake(strategy, query, top_k):
        ranked, latency = _RANKINGS[strategy.value]
        return ranked[:top_k], latency

    with patch("app.routers.experiments.run_retriever", new=AsyncMock(side_effect=fake)) as m:
        yield m


class TestCompare:
    def test_defaults_to_dense_bm25_hybrid(self, client, patched_retrievers):
        resp = client.post("/api/experiments/compare", json={"query": "What is RRF?"})

        assert resp.status_code == 200
        body = resp.json()
        assert [r["strategy"] for r in body["results"]] == ["dense", "bm25", "hybrid"]
        assert body["top_k"] == 5
        bm25 = body["results"][1]
        assert bm25["status"] == "completed"
        assert bm25["latency_ms"] == 15.0
        assert bm25["score_label"] == "BM25 score"
        assert [c["chunk_id"] for c in bm25["chunks"]] == [3, 1]
        assert bm25["chunks"][0] == {
            "rank": 1,
            "chunk_id": 3,
            "file_name": "paper.pdf",
            "title": "paper.pdf",
            "page_number": 4,
            "chunk_index": 3,
            "score": 7.5,
            "content": "chunk 3",
        }

    def test_persists_one_run_per_strategy(self, client, db_session, patched_retrievers):
        client.post(
            "/api/experiments/compare",
            json={"query": "q", "strategies": ["dense", "bm25"], "top_k": 1},
        )

        runs = db_session.query(ExperimentRun).order_by(ExperimentRun.strategy).all()
        assert [r.strategy for r in runs] == ["bm25", "dense"]
        assert runs[0].results == [
            {"chunk_id": 3, "file_name": "paper.pdf", "page_number": 4, "score": 7.5}
        ]
        assert all(r.top_k == 1 and r.status == "completed" for r in runs)

    def test_duplicate_strategies_run_once(self, client, patched_retrievers):
        resp = client.post(
            "/api/experiments/compare",
            json={"query": "q", "strategies": ["bm25", "bm25", "dense"]},
        )
        assert [r["strategy"] for r in resp.json()["results"]] == ["bm25", "dense"]

    def test_failed_strategy_is_reported_not_dropped(self, client, db_session):
        async def fake(strategy, query, top_k):
            if strategy.value == "bm25":
                raise RuntimeError("corpus unavailable")
            ranked, latency = _RANKINGS[strategy.value]
            return ranked, latency

        with patch("app.routers.experiments.run_retriever", new=AsyncMock(side_effect=fake)):
            resp = client.post("/api/experiments/compare", json={"query": "q"})

        assert resp.status_code == 200
        bm25 = resp.json()["results"][1]
        assert bm25["status"] == "failed"
        assert bm25["chunks"] == []
        assert bm25["latency_ms"] is None
        assert "corpus unavailable" in bm25["error_message"]
        failed = db_session.query(ExperimentRun).filter_by(strategy="bm25").one()
        assert failed.status == "failed" and failed.results is None

    @pytest.mark.parametrize(
        "payload",
        [
            {"query": "   "},
            {"query": "q", "top_k": 0},
            {"query": "q", "top_k": 21},
            {"query": "q", "strategies": []},
            {"query": "q", "strategies": ["splade"]},
        ],
    )
    def test_rejects_invalid_requests(self, client, patched_retrievers, payload):
        assert client.post("/api/experiments/compare", json=payload).status_code == 422

    def test_unknown_config_is_404(self, client, patched_retrievers):
        resp = client.post("/api/experiments/compare", json={"query": "q", "config_id": "nope"})
        assert resp.status_code == 404


class TestConfigs:
    def test_create_list_and_run_saved_config(self, client, db_session, patched_retrievers):
        created = client.post(
            "/api/experiments/configs",
            json={"name": "keyword-vs-hybrid", "strategies": ["bm25", "hybrid"], "top_k": 2},
        )
        assert created.status_code == 201
        config = created.json()
        assert config["strategies"] == ["bm25", "hybrid"]

        listed = client.get("/api/experiments/configs").json()
        assert [c["id"] for c in listed] == [config["id"]]

        resp = client.post(
            "/api/experiments/compare", json={"query": "q", "config_id": config["id"]}
        ).json()
        assert [r["strategy"] for r in resp["results"]] == ["bm25", "hybrid"]
        assert resp["top_k"] == 2
        assert resp["config_id"] == config["id"]
        assert {r.config_id for r in db_session.query(ExperimentRun).all()} == {config["id"]}

    def test_request_values_override_saved_config(self, client, patched_retrievers):
        config = client.post(
            "/api/experiments/configs", json={"name": "c", "strategies": ["bm25"], "top_k": 2}
        ).json()
        resp = client.post(
            "/api/experiments/compare",
            json={"query": "q", "config_id": config["id"], "strategies": ["dense"], "top_k": 1},
        ).json()
        assert [r["strategy"] for r in resp["results"]] == ["dense"]
        assert resp["top_k"] == 1

    def test_duplicate_name_is_409(self, client):
        client.post("/api/experiments/configs", json={"name": "same"})
        assert client.post("/api/experiments/configs", json={"name": "same"}).status_code == 409

    def test_delete_keeps_runs_but_unlinks_them(self, client, db_session, patched_retrievers):
        config = client.post("/api/experiments/configs", json={"name": "tmp"}).json()
        client.post("/api/experiments/compare", json={"query": "q", "config_id": config["id"]})

        assert client.delete(f"/api/experiments/configs/{config['id']}").status_code == 204
        assert client.get("/api/experiments/configs").json() == []
        runs = db_session.query(ExperimentRun).all()
        assert runs and all(r.config_id is None for r in runs)

    def test_delete_unknown_is_404(self, client):
        assert client.delete("/api/experiments/configs/nope").status_code == 404


class TestRuns:
    def test_lists_recent_runs_newest_first(self, client, patched_retrievers):
        client.post("/api/experiments/compare", json={"query": "first", "strategies": ["dense"]})
        client.post("/api/experiments/compare", json={"query": "second", "strategies": ["bm25"]})

        runs = client.get("/api/experiments/runs").json()
        assert [r["query"] for r in runs] == ["second", "first"]
        assert runs[0]["top_chunk_ids"] == [3, 1]
        assert runs[0]["num_results"] == 2

    def test_limit(self, client, patched_retrievers):
        client.post("/api/experiments/compare", json={"query": "q"})
        assert len(client.get("/api/experiments/runs?limit=2").json()) == 2
