"""Tests for backend/app/services.py."""

import pytest
from unittest.mock import MagicMock, patch

from app.services import generate_embeddings_batch


class TestGenerateEmbeddingsBatch:
    """Tests for batch embedding generation."""

    def test_empty_list_returns_empty(self):
        assert generate_embeddings_batch([]) == []

    @patch("app.services.client")
    def test_standard_batch(self, mock_client):
        """Tests that a batch under the size limit works correctly."""
        # Setup mock response
        mock_response = MagicMock()
        
        mock_emb1 = MagicMock()
        mock_emb1.values = [0.1, 0.2, 0.3]
        mock_emb2 = MagicMock()
        mock_emb2.values = [0.4, 0.5, 0.6]
        
        mock_response.embeddings = [mock_emb1, mock_emb2]
        mock_client.models.embed_content.return_value = mock_response

        # Execute
        texts = ["text1", "text2"]
        result = generate_embeddings_batch(texts, batch_size=100)

        # Assertions
        assert len(result) == 2
        assert result[0] == [0.1, 0.2, 0.3]
        assert result[1] == [0.4, 0.5, 0.6]
        
        # Verify mock was called with correct inputs
        mock_client.models.embed_content.assert_called_once()
        call_kwargs = mock_client.models.embed_content.call_args.kwargs
        assert call_kwargs["contents"] == texts

    @patch("app.services.client")
    def test_batch_splitting(self, mock_client):
        """Tests that inputs larger than batch_size are split into multiple calls."""
        
        # We need a side_effect to return correct mock responses based on batch size
        def mock_embed_content(model, contents, config):
            mock_resp = MagicMock()
            # Return an embedding for each item in contents
            mock_embeddings = []
            for _ in contents:
                emb = MagicMock()
                emb.values = [0.99]
                mock_embeddings.append(emb)
            mock_resp.embeddings = mock_embeddings
            return mock_resp
            
        mock_client.models.embed_content.side_effect = mock_embed_content

        # Execute with 250 texts and batch_size 100
        texts = [f"text_{i}" for i in range(250)]
        result = generate_embeddings_batch(texts, batch_size=100)

        assert len(result) == 250
        assert mock_client.models.embed_content.call_count == 3
        
        # Check call arguments lengths (100, 100, 50)
        calls = mock_client.models.embed_content.call_args_list
        assert len(calls[0].kwargs["contents"]) == 100
        assert len(calls[1].kwargs["contents"]) == 100
        assert len(calls[2].kwargs["contents"]) == 50

    @patch("app.services.client")
    def test_raises_on_missing_embeddings(self, mock_client):
        """Tests that a clear error is raised if lengths mismatch."""
        mock_response = MagicMock()
        mock_emb = MagicMock()
        mock_emb.values = [0.1]
        
        # Input has 2 texts, but mock only returns 1 embedding
        mock_response.embeddings = [mock_emb]
        mock_client.models.embed_content.return_value = mock_response

        with pytest.raises(ValueError, match="Expected 2, got 1"):
            generate_embeddings_batch(["t1", "t2"], batch_size=100)
            
    @patch("app.services.client")
    def test_raises_on_empty_embeddings(self, mock_client):
        """Tests that a clear error is raised if embeddings list is None/empty."""
        mock_response = MagicMock()
        mock_response.embeddings = []
        mock_client.models.embed_content.return_value = mock_response

        with pytest.raises(ValueError, match="Failed to generate embeddings"):
            generate_embeddings_batch(["t1"], batch_size=100)


# ---------------------------------------------------------------------------
# search_documents — dense retrieval with pgvector
# ---------------------------------------------------------------------------

from app.services import search_documents, EMBEDDING_DIM


def _make_db_row(id, title, content, file_name, page_number, chunk_index, distance):
    """Creates a MagicMock that behaves like a SQLAlchemy Row."""
    row = MagicMock()
    row.id = id
    row.title = title
    row.content = content
    row.file_name = file_name
    row.page_number = page_number
    row.chunk_index = chunk_index
    row.distance = distance
    return row


def _make_query_vector(dim=EMBEDDING_DIM, value=0.01):
    """Creates a dummy query vector of the correct dimensionality."""
    return [value] * dim


class TestSearchDocuments:
    """Tests for dense retrieval via search_documents."""

    def _mock_db(self, rows):
        """Returns a mock Session whose .execute().fetchall() yields rows."""
        db = MagicMock()
        db.execute.return_value.fetchall.return_value = rows
        return db

    def test_returns_documents_with_distance(self):
        """Basic retrieval: one row → one (Document, distance) pair."""
        row = _make_db_row(
            id=1, title="T1", content="Hello world",
            file_name="a.pdf", page_number=1, chunk_index=0, distance=0.12,
        )
        db = self._mock_db([row])

        results = search_documents(db, _make_query_vector(), top_k=3)

        assert len(results) == 1
        doc, dist = results[0]
        assert doc.id == 1
        assert doc.content == "Hello world"
        assert doc.file_name == "a.pdf"
        assert doc.page_number == 1
        assert doc.chunk_index == 0
        assert dist == pytest.approx(0.12)

    def test_preserves_all_metadata_fields(self):
        """Verifies that every metadata field round-trips through the function."""
        row = _make_db_row(
            id=42, title="Report (Page 3, Chunk 7)",
            content="Some PDF text here",
            file_name="report.pdf", page_number=3, chunk_index=7, distance=0.05,
        )
        db = self._mock_db([row])

        results = search_documents(db, _make_query_vector(), top_k=1)

        doc, dist = results[0]
        assert doc.id == 42
        assert doc.title == "Report (Page 3, Chunk 7)"
        assert doc.content == "Some PDF text here"
        assert doc.file_name == "report.pdf"
        assert doc.page_number == 3
        assert doc.chunk_index == 7
        assert dist == pytest.approx(0.05)

    def test_results_ordered_by_distance(self):
        """Results come back in ascending distance (most similar first)."""
        rows = [
            _make_db_row(1, "T1", "c1", "a.pdf", 1, 0, 0.05),
            _make_db_row(2, "T2", "c2", "a.pdf", 1, 1, 0.15),
            _make_db_row(3, "T3", "c3", "a.pdf", 2, 2, 0.30),
        ]
        db = self._mock_db(rows)

        results = search_documents(db, _make_query_vector(), top_k=3)

        distances = [dist for _, dist in results]
        assert distances == sorted(distances)

    def test_top_k_limits_result_count(self):
        """SQL LIMIT :top_k should restrict how many rows are returned."""
        rows = [_make_db_row(i, f"T{i}", f"c{i}", "f.pdf", 1, i, 0.1 * i) for i in range(2)]
        db = self._mock_db(rows)

        results = search_documents(db, _make_query_vector(), top_k=2)
        assert len(results) == 2

        # Verify top_k was actually passed to the SQL query
        call_args = db.execute.call_args
        assert call_args[0][1]["top_k"] == 2

    def test_empty_result_returns_empty_list(self):
        """No matching rows → empty list, not an error."""
        db = self._mock_db([])
        results = search_documents(db, _make_query_vector(), top_k=5)
        assert results == []

    # --- Validation tests ---

    def test_raises_on_top_k_zero(self):
        db = self._mock_db([])
        with pytest.raises(ValueError, match="top_k must be an integer between 1 and 100"):
            search_documents(db, _make_query_vector(), top_k=0)

    def test_raises_on_top_k_negative(self):
        db = self._mock_db([])
        with pytest.raises(ValueError, match="top_k must be an integer between 1 and 100"):
            search_documents(db, _make_query_vector(), top_k=-5)

    def test_raises_on_top_k_over_limit(self):
        db = self._mock_db([])
        with pytest.raises(ValueError, match="top_k must be an integer between 1 and 100"):
            search_documents(db, _make_query_vector(), top_k=101)

    def test_raises_on_wrong_vector_dimensions(self):
        db = self._mock_db([])
        bad_vector = [0.1] * 512  # 512 != 768
        with pytest.raises(ValueError, match="must have 768 dimensions"):
            search_documents(db, bad_vector, top_k=3)

    def test_default_top_k_is_3(self):
        """Calling without top_k should default to 3."""
        db = self._mock_db([])
        search_documents(db, _make_query_vector())
        call_args = db.execute.call_args
        assert call_args[0][1]["top_k"] == 3
