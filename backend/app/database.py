import logging
import os

from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

logger = logging.getLogger("ragops.database")

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def add_missing_columns(bind: Engine = engine) -> list[str]:
    """Adds model columns that existing tables lack, and returns what it added.

    create_all creates missing tables but never alters existing ones, so a
    database created before a nullable column was added to a model would
    otherwise fail on every query touching that column. This covers exactly
    that case and nothing more: it only adds nullable columns without
    defaults, which is safe on a live table. Anything else is a real migration
    and is logged rather than attempted.
    """
    inspector = inspect(bind)
    existing_tables = set(inspector.get_table_names())
    added: list[str] = []

    with bind.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue
            present = {col["name"] for col in inspector.get_columns(table.name)}
            for column in table.columns:
                if column.name in present:
                    continue
                if not column.nullable or column.server_default is not None:
                    logger.error(
                        "Column %s.%s is missing and cannot be added automatically; "
                        "migrate this table manually.",
                        table.name, column.name,
                    )
                    continue
                col_type = column.type.compile(dialect=bind.dialect)
                conn.execute(
                    text(f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {col_type}')
                )
                added.append(f"{table.name}.{column.name}")

    if added:
        logger.info("Added missing columns: %s", ", ".join(added))
    return added
