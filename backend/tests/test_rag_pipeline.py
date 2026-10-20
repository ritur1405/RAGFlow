"""Tests for backend/app/rag_pipeline.py — full RAG orchestration.

All external calls (embedding, database search, LLM) are mocked.
No paid API calls are made.
"""

import pytest
from unittest.mock import MagicMock, patch

from app.rag_pipeline import query_rag, RAGResult, SourceChunk
from app.retrieval import RetrievedChunk
from app.config import NO_ANSWER_MESSAGE


def _make_retrieved_chunk(**overrides) -> RetrievedChunk:
    """Factory for RetrievedChunk test objects."""
    defaults = dict(
        chunk_id=1,
        document_name="doc.pdf",
        content="Test chunk content.",
        page_number=1,
        chunk_index=0,
        similarity_score=0.25,
    )
    defaults.update(overrides)
    return RetrievedChunk(**defaults)


class TestQueryRag:
    """Tests for the full RAG pipeline orchestrator."""

    @patch("app.rag_pipeline.generate_answer")
    @patch("app.rag_pipeline.retrieve_chunks")
    def test_full_pipeline_success(self, mock_retrieve, mock_llm):
        """Happy path: retrieval → context → prompt → LLM → answer + sources."""
        chunks = [
            _make_retrieved_chunk(chunk_id=1, content="AI is cool.", chunk_index=0),
            _make_retrieved_chunk(chunk_id=2, content="ML is a subset.", chunk_index=1),
        ]
        mock_retrieve.return_value = chunks
        mock_llm.return_value = "AI is cool and ML is a subset of it. [Source 1] [Source 2]"

        db = MagicMock()
        result = query_rag(db, "What is AI?")

        assert isinstance(result, RAGResult)
        assert "AI is cool" in result.answer
        assert result.query == "What is AI?"
        assert result.num_chunks_retrieved == 2
        assert len(result.sources) == 2

    @patch("app.rag_pipeline.generate_answer")
    @patch("app.rag_pipeline.retrieve_chunks")
    def test_sources_match_retrieved_chunks(self, mock_retrieve, mock_llm):
        """Citations must correspond to actual retrieved chunks — no fabrication."""
        chunk = _make_retrieved_chunk(
            chunk_id=42,
            document_name="report.pdf",
            content="Important finding.",
            page_number=3,
            chunk_index=7,
            similarity_score=0.15,
        )
        mock_retrieve.return_value = [chunk]
        mock_llm.return_value = "The important finding is..."

        db = MagicMock()
        result = query_rag(db, "findings?")

        assert len(result.sources) == 1
        src = result.sources[0]
        assert isinstance(src, SourceChunk)
        assert src.chunk_id == 42
        assert src.document_name == "report.pdf"
        assert src.text == "Important finding."
        assert src.page_number == 3
        assert src.chunk_index == 7
        assert src.score == 0.15

    @patch("app.rag_pipeline.retrieve_chunks")
    def test_no_results_returns_no_answer(self, mock_retrieve):
        """When retrieval returns empty, answer should be the no-answer message."""
        mock_retrieve.return_value = []

        db = MagicMock()
        result = query_rag(db, "Unknown topic?")

        assert result.answer == NO_ANSWER_MESSAGE
        assert result.sources == []
        assert result.num_chunks_retrieved == 0

    def test_empty_question_raises_value_error(self):
        db = MagicMock()
        with pytest.raises(ValueError, match="non-empty"):
            query_rag(db, "")

    def test_whitespace_question_raises_value_error(self):
        db = MagicMock()
        with pytest.raises(ValueError, match="non-empty"):
            query_rag(db, "   ")

    @patch("app.rag_pipeline.retrieve_chunks")
    def test_retrieval_error_returns_error_message(self, mock_retrieve):
        """Retrieval failures should return an error message, not crash."""
        mock_retrieve.side_effect = RuntimeError("Embedding API down")

        db = MagicMock()
        result = query_rag(db, "query?")

        assert "Retrieval error" in result.answer
        assert result.sources == []

    @patch("app.rag_pipeline.generate_answer")
    @patch("app.rag_pipeline.retrieve_chunks")
    def test_llm_error_returns_error_with_sources(self, mock_retrieve, mock_llm):
        """LLM failures should return an error but still include the retrieved sources."""
        mock_retrieve.return_value = [_make_retrieved_chunk()]
        mock_llm.side_effect = RuntimeError("LLM API timeout")

        db = MagicMock()
        result = query_rag(db, "query?")

        assert "Generation error" in result.answer
        # Sources should still be included for transparency
        assert len(result.sources) == 1
        assert result.num_chunks_retrieved == 1

    @patch("app.rag_pipeline.generate_answer")
    @patch("app.rag_pipeline.retrieve_chunks")
    def test_custom_top_k(self, mock_retrieve, mock_llm):
        """Custom top_k should be forwarded to retrieve_chunks."""
        mock_retrieve.return_value = []

        db = MagicMock()
        query_rag(db, "query?", top_k=10)

        mock_retrieve.assert_called_once()
        call_kwargs = mock_retrieve.call_args[1]
        assert call_kwargs["top_k"] == 10

    @patch("app.rag_pipeline.generate_answer")
    @patch("app.rag_pipeline.retrieve_chunks")
    def test_context_passed_to_llm(self, mock_retrieve, mock_llm):
        """The LLM should receive a prompt containing the chunk content."""
        mock_retrieve.return_value = [
            _make_retrieved_chunk(content="Specific fact about RAG.")
        ]
        mock_llm.return_value = "Answer based on context."

        db = MagicMock()
        query_rag(db, "What about RAG?")

        # The prompt sent to LLM should contain the chunk content
        prompt_arg = mock_llm.call_args[0][0]
        assert "Specific fact about RAG." in prompt_arg
        assert "What about RAG?" in prompt_arg

    @patch("app.rag_pipeline.generate_answer")
    @patch("app.rag_pipeline.retrieve_chunks")
    def test_prompt_contains_grounding_instructions(self, mock_retrieve, mock_llm):
        """The prompt should instruct the LLM to stay grounded."""
        mock_retrieve.return_value = [_make_retrieved_chunk()]
        mock_llm.return_value = "Grounded answer."

        db = MagicMock()
        query_rag(db, "question?")

        prompt_arg = mock_llm.call_args[0][0]
        assert NO_ANSWER_MESSAGE in prompt_arg
        assert "ONLY" in prompt_arg

    @patch("app.rag_pipeline.generate_answer")
    @patch("app.rag_pipeline.retrieve_chunks")
    def test_multiple_chunks_all_become_sources(self, mock_retrieve, mock_llm):
        """All retrieved chunks should appear in the sources list."""
        chunks = [
            _make_retrieved_chunk(chunk_id=i, content=f"Chunk {i}.", chunk_index=i)
            for i in range(5)
        ]
        mock_retrieve.return_value = chunks
        mock_llm.return_value = "Combined answer."

        db = MagicMock()
        result = query_rag(db, "multi-source query?")

        assert len(result.sources) == 5
        assert result.num_chunks_retrieved == 5
        for i, src in enumerate(result.sources):
            assert src.chunk_id == i
            assert src.text == f"Chunk {i}."
