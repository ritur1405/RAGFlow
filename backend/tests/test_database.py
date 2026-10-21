"""Tests for add_missing_columns in app/database.py, on SQLite."""

from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, inspect, text

from app.database import Base, add_missing_columns
from app.models import EvalRun  # noqa: F401  (registers the table on Base)


def test_adds_new_nullable_columns_to_an_existing_table(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    # An eval_runs table as it existed before the retrieval-metric columns.
    old = MetaData()
    Table(
        "eval_runs", old,
        Column("id", String, primary_key=True),
        Column("rag_config", String, nullable=False),
        Column("dataset_name", String, nullable=False),
        Column("num_questions", Integer, nullable=False),
        Column("status", String, nullable=False),
    )
    old.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text(
            "INSERT INTO eval_runs VALUES ('r1', 'dense', 'd', 1, 'completed')"
        ))

    added = add_missing_columns(engine)

    assert "eval_runs.avg_ndcg_at_k" in added
    assert "eval_runs.retrieval_k" in added
    columns = {c["name"] for c in inspect(engine).get_columns("eval_runs")}
    assert {"avg_precision_at_k", "avg_recall_at_k", "avg_mrr", "avg_ndcg_at_k"} <= columns
    # Existing rows survive with the new columns null.
    with engine.connect() as conn:
        assert conn.execute(text("SELECT id, avg_mrr FROM eval_runs")).all() == [("r1", None)]
    # Tables that do not exist yet are left to create_all.
    assert "eval_results" not in inspect(engine).get_table_names()
    engine.dispose()


def test_is_a_no_op_when_the_schema_is_current(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'new.db'}")
    Base.metadata.create_all(engine, tables=[EvalRun.__table__])
    assert add_missing_columns(engine) == []
    engine.dispose()


def test_never_adds_a_non_nullable_column(tmp_path, caplog):
    engine = create_engine(f"sqlite:///{tmp_path / 'strict.db'}")
    old = MetaData()
    Table("eval_runs", old, Column("id", String, primary_key=True))
    old.create_all(engine)

    added = add_missing_columns(engine)

    # rag_config is NOT NULL without a default, so it is reported, not added.
    assert "eval_runs.rag_config" not in added
    assert "cannot be added automatically" in caplog.text
    engine.dispose()
