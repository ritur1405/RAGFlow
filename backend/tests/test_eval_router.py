"""Tests for the generation-evaluation API in app/routers/eval.py.

Covers the success path, database persistence, and -- most importantly -- that
every failure mode is represented as a failure rather than as plausible-looking
metrics.

The RAG pipeline and the LLM judge are patched at the router boundary. A
file-backed SQLite database provides real persistence; only the evaluation
tables are created, since `documents` carries a pgvector column SQLite cannot
express.
"""

from app.models import utcnow
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base, get_db
from app.eval_generation import (
    GenerationEvalResult,
    JudgeCallError,
    JudgeScore,
    JudgeUnavailableError,
)
from app.main import app
from app.models import EvalResult, EvalRun
from app.rag_pipelines import PipelineResult
from app.models import Document


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def db_session(tmp_path: Path):
    """Real SQLite persistence for the two evaluation tables."""
    engine = create_engine(f"sqlite:///{tmp_path / 'eval.db'}")
    Base.metadata.create_all(
        bind=engine, tables=[EvalRun.__table__, EvalResult.__table__]
    )
    Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


# POST /api/eval/run is API-key guarded. Tests configure a known key rather
# than bypassing the dependency, so the real auth path is exercised.
EVAL_KEY = "test-eval-key"
AUTH_HEADER = {"X-API-Key": EVAL_KEY}


@pytest.fixture
def configured_key(monkeypatch):
    monkeypatch.setenv("EVAL_API_KEY", EVAL_KEY)
    return EVAL_KEY


@pytest.fixture
def client(db_session, configured_key):
    """TestClient wired to the SQLite session, sending a valid API key by default."""
    app.dependency_overrides[get_db] = lambda: db_session
    # The app's lifespan would run create_all against the real DATABASE_URL,
    # so it is deliberately not started here.
    with TestClient(app, headers=AUTH_HEADER) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def anon_client(db_session):
    """Same wiring, but sends no API key and configures none on the server."""
    app.dependency_overrides[get_db] = lambda: db_session
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


def _pipeline_result() -> PipelineResult:
    return PipelineResult(
        answer="Revenue grew 12%. [1]",
        context_docs=[
            Document(id=1, title="r.pdf", content="Revenue grew 12% in Q3."),
            Document(id=2, title="r.pdf", content="Costs were flat."),
        ],
        cited_chunk_ids=["1"],
        latency_ms=250.0,
    )


def _eval_result(
    faith=0.9, ans=0.8, ctx=0.7, cite=0.6, overall=0.79, latency=400.0
) -> GenerationEvalResult:
    mk = lambda name, score: JudgeScore(  # noqa: E731
        metric=name, score=score, reasoning=f"{name} reasoning", latency_ms=100.0
    )
    return GenerationEvalResult(
        faithfulness=mk("faithfulness", faith),
        answer_relevance=mk("answer_relevance", ans),
        context_relevance=mk("context_relevance", ctx),
        citation_accuracy=mk("citation_accuracy", cite),
        overall_score=overall,
        total_latency_ms=latency,
    )


@pytest.fixture
def patched_pipeline_and_judge():
    """Patches both external boundaries with successful, distinguishable values."""
    with patch("app.routers.eval.judge_available", return_value=True), \
         patch("app.routers.eval.run_pipeline", new_callable=AsyncMock) as pipe, \
         patch("app.routers.eval.run_generation_eval", new_callable=AsyncMock) as judge:
        pipe.return_value = _pipeline_result()
        judge.return_value = _eval_result()
        yield pipe, judge


_PAYLOAD = {
    "dataset_name": "smoke-set",
    "questions": [{"question": "How did revenue change?", "ground_truth": "Up 12%"}],
    "rag_configs": ["dense"],
}


# --------------------------------------------------------------------------- #
# Success path and persistence
# --------------------------------------------------------------------------- #


class TestRunEvaluationSuccess:
    def test_returns_completed_run_with_measured_metrics(
        self, client, patched_pipeline_and_judge
    ):
        resp = client.post("/api/eval/run", json=_PAYLOAD)

        assert resp.status_code == 201
        runs = resp.json()["runs"]
        assert len(runs) == 1
        run = runs[0]
        assert run["status"] == "completed"
        assert run["error_message"] is None
        assert run["rag_config"] == "dense"
        assert run["rag_config_label"] == "Dense RAG"
        # Averages come from the judge, not from any constant in the router.
        assert run["avg_faithfulness"] == pytest.approx(0.9)
        assert run["avg_answer_relevance"] == pytest.approx(0.8)
        assert run["avg_context_relevance"] == pytest.approx(0.7)
        assert run["avg_citation_accuracy"] == pytest.approx(0.6)
        assert run["avg_overall_score"] == pytest.approx(0.79)
        # Latency is the pipeline's, not a simulated sleep.
        assert run["avg_latency_ms"] == pytest.approx(250.0)

    def test_pipeline_receives_the_requested_config(
        self, client, patched_pipeline_and_judge
    ):
        pipe, _ = patched_pipeline_and_judge
        client.post("/api/eval/run", json={**_PAYLOAD, "rag_configs": ["hybrid"]})

        config_arg = pipe.await_args.args[0]
        assert config_arg.value == "hybrid"

    def test_judge_receives_the_real_retrieved_context(
        self, client, patched_pipeline_and_judge
    ):
        _, judge = patched_pipeline_and_judge
        client.post("/api/eval/run", json=_PAYLOAD)

        kwargs = judge.await_args.kwargs
        assert kwargs["answer"] == "Revenue grew 12%. [1]"
        assert [c.chunk_id for c in kwargs["context_chunks"]] == ["1", "2"]
        assert kwargs["context_chunks"][0].text == "Revenue grew 12% in Q3."
        assert kwargs["cited_chunk_ids"] == ["1"]

    def test_all_four_configs_each_execute_the_pipeline(
        self, client, patched_pipeline_and_judge
    ):
        pipe, judge = patched_pipeline_and_judge
        resp = client.post(
            "/api/eval/run",
            json={
                **_PAYLOAD,
                "rag_configs": ["baseline", "dense", "hybrid", "hybrid_reranker"],
            },
        )

        assert resp.status_code == 201
        assert len(resp.json()["runs"]) == 4
        assert pipe.await_count == 4
        assert judge.await_count == 4
        assert [c.args[0].value for c in pipe.await_args_list] == [
            "baseline", "dense", "hybrid", "hybrid_reranker",
        ]

    def test_defaults_to_all_configs_when_unspecified(
        self, client, patched_pipeline_and_judge
    ):
        pipe, _ = patched_pipeline_and_judge
        resp = client.post(
            "/api/eval/run",
            json={"dataset_name": "d", "questions": [{"question": "q"}]},
        )

        assert resp.status_code == 201
        assert pipe.await_count == 4

    def test_averages_across_multiple_questions(self, client):
        with patch("app.routers.eval.judge_available", return_value=True), \
             patch("app.routers.eval.run_pipeline", new_callable=AsyncMock) as pipe, \
             patch("app.routers.eval.run_generation_eval", new_callable=AsyncMock) as judge:
            pipe.return_value = _pipeline_result()
            judge.side_effect = [
                _eval_result(faith=1.0, overall=1.0),
                _eval_result(faith=0.0, overall=0.0),
            ]
            resp = client.post(
                "/api/eval/run",
                json={
                    "dataset_name": "d",
                    "questions": [{"question": "q1"}, {"question": "q2"}],
                    "rag_configs": ["dense"],
                },
            )

        run = resp.json()["runs"][0]
        assert run["avg_faithfulness"] == pytest.approx(0.5)
        assert run["avg_overall_score"] == pytest.approx(0.5)


class TestPersistence:
    def test_per_question_rows_are_stored_with_reasoning(
        self, client, db_session, patched_pipeline_and_judge
    ):
        run_id = client.post("/api/eval/run", json=_PAYLOAD).json()["runs"][0]["run_id"]

        rows = db_session.query(EvalResult).filter_by(run_id=run_id).all()
        assert len(rows) == 1
        row = rows[0]
        assert row.question == "How did revenue change?"
        assert row.ground_truth == "Up 12%"
        assert row.generated_answer == "Revenue grew 12%. [1]"
        assert row.context_chunk_ids == ["1", "2"]
        assert row.cited_chunk_ids == ["1"]
        assert row.faithfulness_reasoning == "faithfulness reasoning"
        # Pipeline and judge latency are recorded separately, not conflated.
        assert row.pipeline_latency_ms == pytest.approx(250.0)
        assert row.judge_latency_ms == pytest.approx(400.0)

    def test_results_endpoint_returns_stored_detail(
        self, client, patched_pipeline_and_judge
    ):
        run_id = client.post("/api/eval/run", json=_PAYLOAD).json()["runs"][0]["run_id"]

        resp = client.get(f"/api/eval/results/{run_id}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["run"]["run_id"] == run_id
        assert len(body["results"]) == 1
        detail = body["results"][0]
        assert detail["faithfulness"]["score"] == pytest.approx(0.9)
        assert detail["faithfulness"]["reasoning"] == "faithfulness reasoning"
        assert detail["cited_chunk_ids"] == ["1"]

    def test_unknown_run_id_is_404(self, client):
        assert client.get("/api/eval/results/does-not-exist").status_code == 404


# --------------------------------------------------------------------------- #
# Failure modes -- the core of the no-fabricated-metrics guarantee
# --------------------------------------------------------------------------- #


class TestMissingGeminiConfiguration:
    def test_returns_503_and_creates_no_runs(self, client, db_session):
        with patch("app.routers.eval.judge_available", return_value=False), \
             patch("app.routers.eval.run_pipeline", new_callable=AsyncMock) as pipe:
            resp = client.post("/api/eval/run", json=_PAYLOAD)

        assert resp.status_code == 503
        assert "not configured" in resp.json()["detail"]
        # Refused before doing any work, so nothing partial is persisted.
        pipe.assert_not_awaited()
        assert db_session.query(EvalRun).count() == 0


class TestPipelineFailure:
    def test_marks_run_failed_and_returns_502(self, client, db_session):
        with patch("app.routers.eval.judge_available", return_value=True), \
             patch("app.routers.eval.run_pipeline", new_callable=AsyncMock) as pipe:
            pipe.side_effect = RuntimeError("pgvector unreachable")
            resp = client.post("/api/eval/run", json=_PAYLOAD)

        assert resp.status_code == 502
        assert "RAG pipeline failed" in resp.json()["detail"]

        run = db_session.query(EvalRun).one()
        assert run.status == "failed"
        assert "pgvector unreachable" in run.error_message
        assert run.completed_at is not None
        # No metric survives a failed run, so nothing can be read as a result.
        assert run.avg_faithfulness is None
        assert run.avg_overall_score is None
        assert run.avg_latency_ms is None

    def test_reranker_failure_propagates_as_failed_run(self, client, db_session):
        with patch("app.routers.eval.judge_available", return_value=True), \
             patch("app.routers.eval.run_pipeline", new_callable=AsyncMock) as pipe:
            pipe.side_effect = RuntimeError("LLM reranker failed: rate limited")
            resp = client.post(
                "/api/eval/run", json={**_PAYLOAD, "rag_configs": ["hybrid_reranker"]}
            )

        assert resp.status_code == 502
        assert db_session.query(EvalRun).one().status == "failed"


class TestJudgeFailure:
    def test_judge_call_error_marks_run_failed(self, client, db_session):
        with patch("app.routers.eval.judge_available", return_value=True), \
             patch("app.routers.eval.run_pipeline", new_callable=AsyncMock) as pipe, \
             patch("app.routers.eval.run_generation_eval", new_callable=AsyncMock) as judge:
            pipe.return_value = _pipeline_result()
            judge.side_effect = JudgeCallError("judge 429 after 3 attempts")
            resp = client.post("/api/eval/run", json=_PAYLOAD)

        assert resp.status_code == 502
        run = db_session.query(EvalRun).one()
        assert run.status == "failed"
        assert "judge 429" in run.error_message
        assert run.avg_faithfulness is None

    def test_judge_unavailable_mid_run_returns_503(self, client, db_session):
        with patch("app.routers.eval.judge_available", return_value=True), \
             patch("app.routers.eval.run_pipeline", new_callable=AsyncMock) as pipe, \
             patch("app.routers.eval.run_generation_eval", new_callable=AsyncMock) as judge:
            pipe.return_value = _pipeline_result()
            judge.side_effect = JudgeUnavailableError("key revoked")
            resp = client.post("/api/eval/run", json=_PAYLOAD)

        assert resp.status_code == 503
        assert db_session.query(EvalRun).one().status == "failed"

    def test_no_results_persisted_for_a_failed_run(self, client, db_session):
        with patch("app.routers.eval.judge_available", return_value=True), \
             patch("app.routers.eval.run_pipeline", new_callable=AsyncMock) as pipe, \
             patch("app.routers.eval.run_generation_eval", new_callable=AsyncMock) as judge:
            pipe.return_value = _pipeline_result()
            judge.side_effect = JudgeCallError("boom")
            client.post("/api/eval/run", json=_PAYLOAD)

        assert db_session.query(EvalResult).count() == 0


# --------------------------------------------------------------------------- #
# Comparison endpoint
# --------------------------------------------------------------------------- #


class TestCompare:
    def test_reports_nulls_not_zeros_when_nothing_measured(self, client):
        resp = client.get("/api/eval/compare")

        assert resp.status_code == 200
        configs = resp.json()["configs"]
        assert [c["rag_config"] for c in configs] == [
            "baseline", "dense", "hybrid", "hybrid_reranker",
        ]
        for row in configs:
            assert row["run_id"] is None
            # None means "not measured"; 0.0 would mean "measured as zero".
            assert row["avg_overall_score"] is None
            assert row["num_questions"] == 0

    def test_failed_runs_never_contribute_numbers(self, client, db_session):
        db_session.add(
            EvalRun(
                id="failed-1",
                rag_config="dense",
                dataset_name="d",
                num_questions=1,
                status="failed",
                error_message="pipeline exploded",
                created_at=utcnow(),
                avg_overall_score=None,
            )
        )
        db_session.commit()

        rows = {c["rag_config"]: c for c in client.get("/api/eval/compare").json()["configs"]}

        assert rows["dense"]["run_id"] is None
        assert rows["dense"]["avg_overall_score"] is None

    def test_shows_completed_run(self, client, patched_pipeline_and_judge):
        client.post("/api/eval/run", json=_PAYLOAD)

        rows = {c["rag_config"]: c for c in client.get("/api/eval/compare").json()["configs"]}

        assert rows["dense"]["avg_overall_score"] == pytest.approx(0.79)
        assert rows["dense"]["num_questions"] == 1
        assert rows["baseline"]["avg_overall_score"] is None


# --------------------------------------------------------------------------- #
# Contract and hygiene guards
# --------------------------------------------------------------------------- #


class TestApiContracts:
    def test_week3_routes_still_registered(self):
        paths = set(app.openapi()["paths"])
        for path in (
            "/",
            "/documents/",
            "/documents/upload/",
            "/documents/{doc_id}",
            "/query/",
        ):
            assert path in paths, f"Week 3 contract {path} disappeared"

    def test_week6_eval_routes_registered(self):
        paths = set(app.openapi()["paths"])
        assert "/api/eval/run" in paths
        assert "/api/eval/compare" in paths
        assert "/api/eval/results/{run_id}" in paths

    def test_upload_alias_shares_the_documents_upload_handler(self):
        routes = {
            r.path: r
            for r in app.routes
            if getattr(r, "path", None) in ("/upload/", "/documents/upload/")
        }
        assert set(routes) == {"/upload/", "/documents/upload/"}
        assert routes["/upload/"].endpoint is routes["/documents/upload/"].endpoint

    def test_empty_question_list_is_rejected(self, client):
        resp = client.post(
            "/api/eval/run", json={"dataset_name": "d", "questions": []}
        )
        assert resp.status_code == 422

    def test_unknown_config_is_rejected(self, client):
        resp = client.post(
            "/api/eval/run", json={**_PAYLOAD, "rag_configs": ["magic"]}
        )
        assert resp.status_code == 422


class TestEvalRunAuthorization:
    """POST /api/eval/run must not be callable without authorization.

    Each call spends Gemini credits, so every rejection path is asserted to
    stop before the pipeline is ever invoked.
    """

    @pytest.fixture(autouse=True)
    def _never_runs_the_pipeline(self):
        """Patched for every test here: an unauthorized call must not reach it."""
        with patch("app.routers.eval.judge_available", return_value=True), \
             patch("app.routers.eval.run_pipeline", new_callable=AsyncMock) as pipe, \
             patch("app.routers.eval.run_generation_eval", new_callable=AsyncMock) as judge:
            pipe.return_value = _pipeline_result()
            judge.return_value = _eval_result()
            self.pipe = pipe
            yield

    def test_missing_api_key_is_401(self, anon_client, monkeypatch, db_session):
        monkeypatch.setenv("EVAL_API_KEY", EVAL_KEY)

        resp = anon_client.post("/api/eval/run", json=_PAYLOAD)

        assert resp.status_code == 401
        assert "X-API-Key" in resp.json()["detail"]
        self.pipe.assert_not_awaited()
        assert db_session.query(EvalRun).count() == 0

    def test_wrong_api_key_is_401(self, anon_client, monkeypatch, db_session):
        monkeypatch.setenv("EVAL_API_KEY", EVAL_KEY)

        resp = anon_client.post(
            "/api/eval/run", json=_PAYLOAD, headers={"X-API-Key": "not-the-key"}
        )

        assert resp.status_code == 401
        self.pipe.assert_not_awaited()
        assert db_session.query(EvalRun).count() == 0

    def test_empty_api_key_header_is_401(self, anon_client, monkeypatch):
        monkeypatch.setenv("EVAL_API_KEY", EVAL_KEY)

        resp = anon_client.post(
            "/api/eval/run", json=_PAYLOAD, headers={"X-API-Key": ""}
        )

        assert resp.status_code == 401
        self.pipe.assert_not_awaited()

    def test_endpoint_is_closed_when_server_has_no_key_configured(
        self, anon_client, monkeypatch, db_session
    ):
        """Fail closed: an unconfigured server must not run evaluations at all."""
        monkeypatch.delenv("EVAL_API_KEY", raising=False)

        resp = anon_client.post(
            "/api/eval/run", json=_PAYLOAD, headers={"X-API-Key": "anything"}
        )

        assert resp.status_code == 503
        assert "EVAL_API_KEY is not configured" in resp.json()["detail"]
        self.pipe.assert_not_awaited()
        assert db_session.query(EvalRun).count() == 0

    def test_blank_configured_key_does_not_enable_the_endpoint(
        self, anon_client, monkeypatch
    ):
        """Whitespace is not a key; it must not accidentally open the endpoint."""
        monkeypatch.setenv("EVAL_API_KEY", "   ")

        resp = anon_client.post(
            "/api/eval/run", json=_PAYLOAD, headers={"X-API-Key": "   "}
        )

        assert resp.status_code == 503
        self.pipe.assert_not_awaited()

    def test_valid_api_key_is_accepted(self, anon_client, monkeypatch):
        monkeypatch.setenv("EVAL_API_KEY", EVAL_KEY)

        resp = anon_client.post(
            "/api/eval/run", json=_PAYLOAD, headers={"X-API-Key": EVAL_KEY}
        )

        assert resp.status_code == 201
        self.pipe.assert_awaited()

    def test_read_endpoints_remain_open_without_a_key(self, anon_client, monkeypatch):
        """The dashboard reads results without holding a secret."""
        monkeypatch.setenv("EVAL_API_KEY", EVAL_KEY)

        assert anon_client.get("/api/eval/compare").status_code == 200
        assert anon_client.get("/api/eval/results/unknown-id").status_code == 404

    def test_week3_routes_are_not_key_guarded(self, anon_client, monkeypatch):
        """Guarding must be scoped to evaluation execution, not existing routes."""
        monkeypatch.setenv("EVAL_API_KEY", EVAL_KEY)

        resp = anon_client.get("/")
        assert resp.status_code == 200

    def test_openapi_advertises_the_security_scheme_on_run_only(self):
        spec = app.openapi()
        run_op = spec["paths"]["/api/eval/run"]["post"]
        assert "security" in run_op, "POST /api/eval/run should declare a security scheme"
        assert "security" not in spec["paths"]["/api/eval/compare"]["get"]


class TestStartupWiring:
    """Table creation must happen at startup, not on import.

    Substitutes for the part of the Docker verification that needs a live
    Postgres: it proves the lifespan hook is what triggers create_all, so the
    container can be started before the database is accepting connections.
    """

    def test_importing_the_app_does_not_touch_the_database(self):
        """Already true implicitly -- this test suite imports app.main."""
        import app.main as m

        assert hasattr(m, "lifespan")
        src = Path(m.__file__).read_text(encoding="utf-8")
        create_all_line = next(
            line for line in src.splitlines() if "create_all" in line
        )
        # The only create_all call must be indented inside the lifespan function.
        assert create_all_line.startswith("    "), (
            "create_all should live inside lifespan, not at module level"
        )

    def test_lifespan_creates_tables_on_startup(self):
        """Entering the app's lifespan must invoke create_all against the engine."""
        import app.main as m

        with patch.object(m.Base.metadata, "create_all") as create_all:
            with TestClient(m.app):
                pass

        create_all.assert_called_once()
        assert create_all.call_args.kwargs["bind"] is m.engine

    def test_eval_table_ddl_is_valid_on_a_real_database(self, tmp_path):
        """The evaluation tables' DDL executes for real (SQLite stand-in).

        The `documents` table needs pgvector and so cannot be created here;
        that part is only exercised against Postgres.
        """
        from sqlalchemy import create_engine, inspect

        engine = create_engine(f"sqlite:///{tmp_path / 'ddl.db'}")
        Base.metadata.create_all(
            bind=engine, tables=[EvalRun.__table__, EvalResult.__table__]
        )
        names = set(inspect(engine).get_table_names())
        engine.dispose()

        assert {"eval_runs", "eval_results"} <= names


class TestNoFabricatedMetrics:
    """Guards the Phase 6 requirement at the source level.

    These constants were the hardcoded 'realistic' scores the original Week 6
    code substituted whenever the pipeline or judge failed. If any of them
    reappears in the evaluation path, this fails.
    """

    FORBIDDEN = ["0.88", "0.90", "0.85", "0.95", "0.91", "1.00"]
    MODULES = [
        "backend/app/routers/eval.py",
        "backend/app/eval_generation.py",
        "backend/app/rag_pipelines.py",
    ]

    def test_no_hardcoded_metric_constants(self):
        root = Path(__file__).resolve().parents[2]
        for rel in self.MODULES:
            src = (root / rel).read_text(encoding="utf-8")
            for bad in self.FORBIDDEN:
                assert bad not in src, f"{rel} contains fabricated metric value {bad}"

    def test_no_fallback_pipeline_or_simulated_latency(self):
        root = Path(__file__).resolve().parents[2]
        src = (root / "backend/app/routers/eval.py").read_text(encoding="utf-8")
        assert "_fallback_pipeline" not in src
        assert "simulated_latency" not in src
        assert "asyncio.sleep" not in src

    def test_router_does_not_import_the_nonexistent_run_pipeline_helper(self):
        root = Path(__file__).resolve().parents[2]
        src = (root / "backend/app/routers/eval.py").read_text(encoding="utf-8")
        assert "from app.rag_pipeline import run_pipeline" not in src

    def test_tables_are_not_created_at_import_time(self):
        root = Path(__file__).resolve().parents[2]
        src = (root / "backend/app/routers/eval.py").read_text(encoding="utf-8")
        assert "create_all" not in src
