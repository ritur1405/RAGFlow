"""Tests for backend/app/ingestion.py"""

import pytest
from unittest.mock import MagicMock, patch

from app.ingestion import process_pdf_document


class TestProcessPdfDocument:
    """Tests for the PDF ingestion orchestrator."""

    @patch("app.ingestion.generate_embeddings_batch")
    @patch("app.ingestion.chunk_pages")
    @patch("app.ingestion.clean_text")
    @patch("app.ingestion.extract_text_from_pdf")
    def test_end_to_end_success(
        self, mock_extract, mock_clean, mock_chunk, mock_embed
    ):
        """Tests that the pipeline calls all functions correctly and assembles the result."""
        
        # 1. Mock extraction
        mock_extract.return_value = [{"page_number": 1, "text": "raw text"}]
        
        # 2. Mock cleaning
        mock_clean.return_value = "cleaned text"
        
        # 3. Mock chunking
        mock_chunk.return_value = [
            {"content": "cleaned text", "page_number": 1, "chunk_index": 0}
        ]
        
        # 4. Mock embeddings
        mock_embed.return_value = [[0.5, 0.6, 0.7]]
        
        # Execute
        result = process_pdf_document(b"fake-bytes")
        
        # Verify result structure
        assert len(result) == 1
        assert result[0]["content"] == "cleaned text"
        assert result[0]["page_number"] == 1
        assert result[0]["chunk_index"] == 0
        assert result[0]["embedding"] == [0.5, 0.6, 0.7]
        
        # Verify calls
        mock_extract.assert_called_once_with(b"fake-bytes")
        mock_clean.assert_called_once_with("raw text")
        mock_chunk.assert_called_once_with(
            [{"page_number": 1, "text": "cleaned text"}], 
            chunk_size=500, 
            chunk_overlap=50
        )
        mock_embed.assert_called_once_with(
            ["cleaned text"], 
            task_type="RETRIEVAL_DOCUMENT", 
            batch_size=100
        )

    @patch("app.ingestion.extract_text_from_pdf")
    @patch("app.ingestion.clean_text")
    def test_no_text_remaining(self, mock_clean, mock_extract):
        """Tests that a ValueError is raised if cleaning leaves no text."""
        
        mock_extract.return_value = [{"page_number": 1, "text": "   "}]
        mock_clean.return_value = ""  # Cleaned away all text
        
        with pytest.raises(ValueError, match="No text remaining"):
            process_pdf_document(b"fake-bytes")

    @patch("app.ingestion.generate_embeddings_batch")
    @patch("app.ingestion.chunk_pages")
    @patch("app.ingestion.clean_text")
    @patch("app.ingestion.extract_text_from_pdf")
    def test_embedding_length_mismatch(
        self, mock_extract, mock_clean, mock_chunk, mock_embed
    ):
        """Tests that a mismatch in chunk count vs embedding count raises an error."""
        
        mock_extract.return_value = [{"page_number": 1, "text": "raw text"}]
        mock_clean.return_value = "cleaned text"
        
        # Return 2 chunks
        mock_chunk.return_value = [
            {"content": "chunk1", "page_number": 1, "chunk_index": 0},
            {"content": "chunk2", "page_number": 1, "chunk_index": 1}
        ]
        
        # Return only 1 embedding
        mock_embed.return_value = [[0.1]]
        
        with pytest.raises(ValueError, match="does not match chunk count"):
            process_pdf_document(b"fake-bytes")
