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
