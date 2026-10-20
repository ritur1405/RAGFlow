"""Tests for backend/app/retrieval.py — dense retrieval layer.

All external calls (embedding API, database) are mocked.
"""

import pytest
from unittest.mock import MagicMock, patch

from app.retrieval import retrieve_chunks, RetrievedChunk
from app.models import Document


class TestRetrieveChunks:
    """Tests for the retrieve_chunks function."""

    def _make_doc(self, **overrides) -> Document:
        """Helper to create a Document ORM object for mocking."""
        defaults = dict(
            id=1,
            file_name="test.pdf",
            title="test.pdf (Page 1, Chunk 0)",
            content="Chunk text.",
            page_number=1,
            chunk_index=0,
        )
        defaults.update(overrides)
        return Document(**defaults)

    @patch("app.retrieval.search_documents")
    @patch("app.retrieval.generate_embedding")
    def test_successful_retrieval(self, mock_embed, mock_search):
        """Happy path: query → embedding → search → structured results."""
        mock_embed.return_value = [0.1] * 768

        doc = self._make_doc()
        mock_search.return_value = [(doc, 0.3)]

        db = MagicMock()
        results = retrieve_chunks(db, "What is AI?", top_k=3)

        assert len(results) == 1
        assert isinstance(results[0], RetrievedChunk)
        assert results[0].chunk_id == 1
        assert results[0].document_name == "test.pdf"
        assert results[0].content == "Chunk text."
        assert results[0].similarity_score == 0.3

        # Verify embedding was called with query task type
        mock_embed.assert_called_once_with("What is AI?", task_type="RETRIEVAL_QUERY")

    @patch("app.retrieval.search_documents")
    @patch("app.retrieval.generate_embedding")
    def test_filters_by_distance_threshold(self, mock_embed, mock_search):
        """Chunks beyond max_distance should be excluded."""
        mock_embed.return_value = [0.1] * 768

        doc_close = self._make_doc(id=1, content="Close match.")
        doc_far = self._make_doc(id=2, content="Far match.")
        mock_search.return_value = [
            (doc_close, 0.3),
            (doc_far, 0.9),  # Above default threshold of 0.8
        ]

        db = MagicMock()
        results = retrieve_chunks(db, "query", max_distance=0.8)

        assert len(results) == 1
        assert results[0].chunk_id == 1

    @patch("app.retrieval.search_documents")
    @patch("app.retrieval.generate_embedding")
    def test_returns_empty_when_all_beyond_threshold(self, mock_embed, mock_search):
        """When every chunk exceeds the threshold, return empty list."""
        mock_embed.return_value = [0.1] * 768
        mock_search.return_value = [
            (self._make_doc(), 0.95),
        ]

        db = MagicMock()
        results = retrieve_chunks(db, "query", max_distance=0.8)

        assert results == []

    @patch("app.retrieval.search_documents")
    @patch("app.retrieval.generate_embedding")
    def test_returns_empty_when_no_documents(self, mock_embed, mock_search):
        """When the database has no documents, return empty list."""
        mock_embed.return_value = [0.1] * 768
        mock_search.return_value = []

        db = MagicMock()
        results = retrieve_chunks(db, "query")

        assert results == []

    def test_raises_on_empty_query(self):
        db = MagicMock()
        with pytest.raises(ValueError, match="non-empty"):
            retrieve_chunks(db, "")

    def test_raises_on_whitespace_query(self):
        db = MagicMock()
        with pytest.raises(ValueError, match="non-empty"):
            retrieve_chunks(db, "   ")

    @patch("app.retrieval.generate_embedding")
    def test_raises_runtime_error_on_embedding_failure(self, mock_embed):
        """Embedding API failure should be wrapped in a RuntimeError."""
        mock_embed.side_effect = Exception("API timeout")

        db = MagicMock()
        with pytest.raises(RuntimeError, match="Failed to generate query embedding"):
            retrieve_chunks(db, "query")

    @patch("app.retrieval.search_documents")
    @patch("app.retrieval.generate_embedding")
    def test_raises_runtime_error_on_search_failure(self, mock_embed, mock_search):
        """Database search failure should be wrapped in a RuntimeError."""
        mock_embed.return_value = [0.1] * 768
        mock_search.side_effect = Exception("DB connection lost")

        db = MagicMock()
        with pytest.raises(RuntimeError, match="Vector search failed"):
            retrieve_chunks(db, "query")

    @patch("app.retrieval.search_documents")
    @patch("app.retrieval.generate_embedding")
    def test_preserves_metadata(self, mock_embed, mock_search):
        """All metadata fields should be correctly mapped."""
        mock_embed.return_value = [0.1] * 768

        doc = self._make_doc(
            id=42,
            file_name="report.pdf",
            content="Important content.",
            page_number=5,
            chunk_index=12,
        )
        mock_search.return_value = [(doc, 0.456789)]

        db = MagicMock()
        results = retrieve_chunks(db, "query")

        r = results[0]
        assert r.chunk_id == 42
        assert r.document_name == "report.pdf"
        assert r.content == "Important content."
        assert r.page_number == 5
        assert r.chunk_index == 12
        assert r.similarity_score == 0.456789

    @patch("app.retrieval.search_documents")
    @patch("app.retrieval.generate_embedding")
    def test_respects_custom_top_k(self, mock_embed, mock_search):
        """Custom top_k should be passed to search_documents."""
        mock_embed.return_value = [0.1] * 768
        mock_search.return_value = []

        db = MagicMock()
        retrieve_chunks(db, "query", top_k=10)

        # Verify search was called with top_k=10
        mock_search.assert_called_once()
        call_kwargs = mock_search.call_args
        assert call_kwargs[1]["top_k"] == 10 or call_kwargs[0][2] == 10

    @patch("app.retrieval.search_documents")
    @patch("app.retrieval.generate_embedding")
    def test_handles_missing_file_name(self, mock_embed, mock_search):
        """Documents without file_name should map to None."""
        mock_embed.return_value = [0.1] * 768

        doc = self._make_doc(file_name=None)
        mock_search.return_value = [(doc, 0.3)]

        db = MagicMock()
        results = retrieve_chunks(db, "query")

        assert results[0].document_name is None
