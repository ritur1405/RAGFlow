"""Tests for the retrieval-only benchmark (app/retrieval_benchmark.py and
POST /api/experiments/benchmark).

No Gemini calls and no pgvector: embeddings and retrievers are patched.
Persistence uses a real SQLite file.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app import rag_pipelines as rp
from app import retrieval_benchmark as rb
from app.database import Base, get_db
from app.eval_retrieval import RetrievalScores
from app.main import app
from app.models import Document, RetrievalBenchmarkRun

DATASET = {
    "dataset_name": "tiny",
    "questions": [
        {"question": "q1", "source_file": "paper.pdf", "relevant_pages": [5], "type": "keyword"},
        {"question": "q2", "source_file": "paper.pdf", "relevant_pages": [7], "type": "paraphrase"},
        {"question": "unlabelled", "source_file": "paper.pdf", "relevant_pages": []},
    ],
}


@pytest.fixture(autouse=True)
def _fresh_cache():
    rb.clear_embedding_cache()
    yield
    rb.clear_embedding_cache()


def _doc(doc_id, page):
    return Document(id=doc_id, file_name="paper.pdf", title="t", content="c", page_number=page)


# --------------------------------------------------------------------------- #
# Embedding cache
# --------------------------------------------------------------------------- #


def test_each_question_is_embedded_once_per_process():
    with patch.object(rb, "generate_embedding", return_value=[0.1] * 768) as embed:
        _, first = rb.query_embeddings(["q1", "q2"])
        vectors, second = rb.query_embeddings(["q1", "q2", "q3"])

    assert embed.call_count == 3
    assert (first.calls_made, first.reused) == (2, 0)
    assert (second.calls_made, second.reused) == (1, 2)
    assert set(vectors) == {"q1", "q2", "q3"}
    assert embed.call_args.kwargs["task_type"] == "RETRIEVAL_QUERY"


def test_dense_retrieval_uses_the_supplied_vector_instead_of_embedding():
    with patch.object(rp, "generate_embedding") as embed, \
         patch.object(rp, "search_documents", return_value=[]) as search:
        with rp.precomputed_query_embeddings({"q1": [0.5] * 768}):
            rp.retrieve_dense(MagicMock(), "q1")
            rp.retrieve_dense(MagicMock(), "other question")

    # q1 used the supplied vector; the unknown question still embedded once.
    assert search.call_args_list[0].args[1] == [0.5] * 768
    embed.assert_called_once()


def test_supplied_vectors_do_not_leak_outside_the_block():
    with patch.object(rp, "generate_embedding", return_value=[0.0] * 768) as embed, \
         patch.object(rp, "search_documents", return_value=[]):
        with rp.precomputed_query_embeddings({"q1": [0.5] * 768}):
            pass
        rp.retrieve_dense(MagicMock(), "q1")
    embed.assert_called_once()


# --------------------------------------------------------------------------- #
# Scoring and aggregation
# --------------------------------------------------------------------------- #


def test_run_strategy_scores_each_question_against_its_pages():
    fake = MagicMock(side_effect=[
        [(_doc(1, 5), 1.0), (_doc(2, 9), 0.5)],  # q1: relevant at rank 1
        [(_doc(3, 9), 1.0)],                     # q2: miss
    ])
    with patch.dict(rp.RETRIEVER_REGISTRY, {"bm25": fake}), \
         patch.object(rb, "SessionLocal", return_value=MagicMock()):
        rows = rb.run_strategy("bm25", DATASET["questions"][:2], top_k=3, embeddings={})

    assert [r.retrieved_pages for r in rows] == [[5, 9], [9]]
    assert rows[0].scores.reciprocal_rank == 1.0
    assert rows[1].scores.ndcg_at_k == 0.0
    assert fake.call_args.kwargs["top_k"] == 3


def test_aggregate_overall_and_by_type():
    def row(qtype, rr):
        return rb.QuestionResult("q", qtype, [1], [1], latency_ms=10.0,
                                 scores=RetrievalScores(3, rr / 3, rr, rr, rr))

    summary = rb.aggregate([row("keyword", 1.0), row("keyword", 0.0), row("paraphrase", 0.5)])

    assert summary["overall"]["n"] == 3
    assert summary["overall"]["reciprocal_rank"] == pytest.approx(0.5)
    assert summary["by_type"]["keyword"]["reciprocal_rank"] == pytest.approx(0.5)
    assert summary["by_type"]["paraphrase"]["n"] == 1
    assert summary["avg_latency_ms"] == pytest.approx(10.0)


# --------------------------------------------------------------------------- #
# Endpoint
# --------------------------------------------------------------------------- #


@pytest.fixture
def db_session(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'bench.db'}")
    Base.metadata.create_all(engine, tables=[Document.__table__, RetrievalBenchmarkRun.__table__])
    session = sessionmaker(bind=engine)()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def client(db_session):
    app.dependency_overrides[get_db] = lambda: db_session
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def indexed(db_session):
    db_session.add(_doc(1, 5))
    db_session.commit()


def _fake_run_strategy(strategy, questions, top_k, embeddings):
    hit = strategy != "bm25"  # bm25 misses everything, the rest hit at rank 1
    return [
        rb.QuestionResult(q["question"], q.get("type"), q["relevant_pages"],
                          [q["relevant_pages"][0] if hit else 99], 5.0,
                          RetrievalScores(top_k, 1 / top_k if hit else 0, 1.0 if hit else 0,
                                          1.0 if hit else 0, 0.47 if hit else 0))
        for q in questions
    ]


@pytest.fixture
def patched(indexed):
    with patch("app.routers.experiments.load_benchmark", return_value=DATASET), \
         patch("app.routers.experiments.query_embeddings",
               return_value=({}, rb.EmbeddingStats(calls_made=2, latencies_ms=[100.0, 120.0]))) as emb, \
         patch("app.routers.experiments.run_strategy", side_effect=_fake_run_strategy) as run:
        yield emb, run


def test_benchmark_scores_every_strategy_on_labelled_questions(client, db_session, patched):
    emb, run = patched
    resp = client.post("/api/experiments/benchmark", json={})

    assert resp.status_code == 200
    body = resp.json()
    assert body["num_questions"] == 2  # the unlabelled question is skipped
    assert body["embedding"] == {"calls_made": 2, "reused": 0, "avg_latency_ms": 110.0}
    assert [r["strategy"] for r in body["results"]] == ["baseline", "dense", "bm25", "hybrid", "hybrid_reranker"]
    bm25 = next(r for r in body["results"] if r["strategy"] == "bm25")
    dense = next(r for r in body["results"] if r["strategy"] == "dense")
    assert bm25["avg_mrr"] == 0.0 and dense["avg_mrr"] == 1.0
    assert dense["by_type"]["keyword"]["n"] == 1
    assert len(body["per_question"]["dense"]) == 2
    # Questions are embedded once for the whole batch, not once per strategy.
    emb.assert_called_once()
    assert db_session.query(RetrievalBenchmarkRun).count() == 5
    assert len({r.batch_id for r in db_session.query(RetrievalBenchmarkRun)}) == 1


def test_selected_strategies_and_k(client, patched):
    _, run = patched
    body = client.post("/api/experiments/benchmark", json={"strategies": ["bm25", "hybrid"], "top_k": 5}).json()
    assert [r["strategy"] for r in body["results"]] == ["bm25", "hybrid"]
    assert body["k"] == 5
    assert run.call_args.args[2] == 5


def test_a_failing_strategy_is_reported_not_dropped(client, indexed):
    def flaky(strategy, *args):
        if strategy == "hybrid_reranker":
            raise RuntimeError("Cross-encoder reranker failed: model missing")
        return _fake_run_strategy(strategy, *args)

    with patch("app.routers.experiments.load_benchmark", return_value=DATASET), \
         patch("app.routers.experiments.query_embeddings", return_value=({}, rb.EmbeddingStats())), \
         patch("app.routers.experiments.run_strategy", side_effect=flaky):
        body = client.post("/api/experiments/benchmark", json={}).json()

    failed = next(r for r in body["results"] if r["strategy"] == "hybrid_reranker")
    assert failed["status"] == "failed"
    assert failed["avg_ndcg_at_k"] is None
    assert "model missing" in failed["error_message"]


def test_refuses_when_the_benchmark_document_is_not_indexed(client):
    with patch("app.routers.experiments.load_benchmark", return_value=DATASET):
        resp = client.post("/api/experiments/benchmark", json={})
    assert resp.status_code == 409
    assert "paper.pdf" in resp.json()["detail"]


def test_embedding_quota_is_503(client, indexed):
    from google.genai import errors as genai_errors

    err = genai_errors.ClientError(429, {"error": {"code": 429, "message": "quota", "status": "X"}})
    with patch("app.routers.experiments.load_benchmark", return_value=DATASET), \
         patch("app.routers.experiments.query_embeddings", side_effect=err):
        resp = client.post("/api/experiments/benchmark", json={})
    assert resp.status_code == 503


def test_history_lists_newest_first(client, patched):
    client.post("/api/experiments/benchmark", json={"strategies": ["dense"]})
    client.post("/api/experiments/benchmark", json={"strategies": ["bm25"]})
    runs = client.get("/api/experiments/benchmark/runs").json()
    assert [r["strategy"] for r in runs] == ["bm25", "dense"]
