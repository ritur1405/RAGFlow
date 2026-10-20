-- Runs once, on first initialisation of an empty Postgres data directory.
-- The pgvector image ships the extension's files but does not enable it in any
-- database, so without this the first Base.metadata.create_all() fails with
-- "type \"vector\" does not exist" when it reaches Document.embedding.
CREATE EXTENSION IF NOT EXISTS vector;
