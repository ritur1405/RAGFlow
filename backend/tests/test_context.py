"""Tests for backend/app/context.py — context and prompt construction.

These are pure-function tests with no mocks needed (no external calls).
"""

from app.context import build_context, build_prompt, PROMPT_TEMPLATE
from app.config import NO_ANSWER_MESSAGE
from app.retrieval import RetrievedChunk


# ---------------------------------------------------------------------------
# build_context
# ---------------------------------------------------------------------------


class TestBuildContext:
    """Tests for context construction from retrieved chunks."""

    def _make_chunk(self, **overrides) -> RetrievedChunk:
        defaults = dict(
            chunk_id=1,
            document_name="test.pdf",
            content="Some text content.",
            page_number=1,
            chunk_index=0,
            similarity_score=0.25,
        )
        defaults.update(overrides)
        return RetrievedChunk(**defaults)

    def test_empty_chunks_returns_empty_string(self):
        assert build_context([]) == ""

    def test_single_chunk_format(self):
        chunk = self._make_chunk(
            document_name="report.pdf", page_number=3, chunk_index=7
        )
        result = build_context([chunk])

        assert "[Source 1 | report.pdf | Page 3 | Chunk 7]" in result
        assert "Some text content." in result

    def test_multiple_chunks_numbered_sequentially(self):
        chunks = [
            self._make_chunk(chunk_id=1, content="First chunk."),
            self._make_chunk(chunk_id=2, content="Second chunk."),
            self._make_chunk(chunk_id=3, content="Third chunk."),
        ]
        result = build_context(chunks)

        assert "[Source 1" in result
        assert "[Source 2" in result
        assert "[Source 3" in result
        assert "First chunk." in result
        assert "Second chunk." in result
        assert "Third chunk." in result

    def test_preserves_retrieval_order(self):
        chunks = [
            self._make_chunk(chunk_id=1, content="First."),
            self._make_chunk(chunk_id=2, content="Second."),
        ]
        result = build_context(chunks)

        # Source 1 should appear before Source 2
        assert result.index("[Source 1") < result.index("[Source 2")
        assert result.index("First.") < result.index("Second.")

    def test_handles_missing_document_name(self):
        chunk = self._make_chunk(document_name=None)
        result = build_context([chunk])

        # Should not contain None or crash
        assert "None" not in result
        assert "[Source 1 | Page 1 | Chunk 0]" in result

    def test_handles_missing_page_and_chunk(self):
        chunk = self._make_chunk(
            document_name="file.pdf", page_number=None, chunk_index=None
        )
        result = build_context([chunk])

        assert "[Source 1 | file.pdf]" in result
        assert "Page" not in result
        assert "Chunk" not in result

    def test_chunks_separated_by_blank_lines(self):
        chunks = [
            self._make_chunk(chunk_id=1, content="A."),
            self._make_chunk(chunk_id=2, content="B."),
        ]
        result = build_context(chunks)

        # Two chunks should be separated by a double newline
        assert "\n\n" in result


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    """Tests for prompt construction."""

    def test_contains_context_and_question(self):
        prompt = build_prompt("Some context here.", "What is X?")

        assert "Some context here." in prompt
        assert "What is X?" in prompt

    def test_contains_grounding_instructions(self):
        prompt = build_prompt("ctx", "q")

        # Key grounding phrases from the template
        assert "ONLY" in prompt
        assert "do not" in prompt.lower() or "Do not" in prompt
        assert NO_ANSWER_MESSAGE in prompt

    def test_contains_citation_instructions(self):
        prompt = build_prompt("ctx", "q")

        assert "Source" in prompt or "source" in prompt

    def test_no_answer_message_is_embedded(self):
        prompt = build_prompt("ctx", "q")

        assert NO_ANSWER_MESSAGE in prompt

    def test_prompt_template_has_required_placeholders(self):
        assert "{context}" in PROMPT_TEMPLATE
        assert "{question}" in PROMPT_TEMPLATE
        assert "{no_answer}" in PROMPT_TEMPLATE
