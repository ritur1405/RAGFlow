"""Unit tests for build_context — the RAG context-builder function.

Tests are pure-logic only (no LLM, no database).  Document-like objects
are created via SimpleNamespace to avoid SQLAlchemy instrumentation issues.
"""

import types as _types
import pytest

from app.services import build_context, ContextResult, SourceMeta, _build_chunk_label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _doc(id, content, *, file_name=None, page_number=None, chunk_index=None, title=None):
    """Creates a Document-like object for testing."""
    return _types.SimpleNamespace(
        id=id,
        title=title or f"Doc {id}",
        content=content,
        file_name=file_name,
        page_number=page_number,
        chunk_index=chunk_index,
    )


# ---------------------------------------------------------------------------
# _build_chunk_label
# ---------------------------------------------------------------------------

class TestBuildChunkLabel:
    def test_full_metadata(self):
        doc = _doc(1, "x", file_name="a.pdf", page_number=2, chunk_index=3)
        assert _build_chunk_label(doc) == 'File: "a.pdf", Page 2, Chunk 3'

    def test_file_name_only(self):
        doc = _doc(1, "x", file_name="b.pdf")
        assert _build_chunk_label(doc) == 'File: "b.pdf"'

    def test_page_and_chunk_no_file(self):
        doc = _doc(1, "x", page_number=5, chunk_index=0)
        assert _build_chunk_label(doc) == "Page 5, Chunk 0"

    def test_no_metadata_falls_back_to_id(self):
        doc = _doc(42, "x")
        assert _build_chunk_label(doc) == "Source ID 42"

    def test_chunk_index_zero_is_included(self):
        """chunk_index=0 is valid metadata, not 'missing'."""
        doc = _doc(1, "x", chunk_index=0)
        assert "Chunk 0" in _build_chunk_label(doc)


# ---------------------------------------------------------------------------
# build_context — empty input
# ---------------------------------------------------------------------------

class TestBuildContextEmpty:
    def test_empty_list_returns_empty_result(self):
        result = build_context([])
        assert isinstance(result, ContextResult)
        assert result.context_text == ""
        assert result.sources == []
        assert result.total_chars == 0
        assert result.chunks_included == 0
        assert result.chunks_dropped == 0


# ---------------------------------------------------------------------------
# build_context — basic formatting
# ---------------------------------------------------------------------------

class TestBuildContextFormatting:
    def test_single_chunk_has_source_label(self):
        doc = _doc(1, "Hello world", file_name="a.pdf", page_number=1, chunk_index=0)
        result = build_context([doc])

        assert "[Source 1" in result.context_text
        assert 'File: "a.pdf"' in result.context_text
        assert "Page 1" in result.context_text
        assert "Chunk 0" in result.context_text
        assert "Hello world" in result.context_text

    def test_multiple_chunks_separated(self):
        docs = [
            _doc(1, "First chunk", file_name="a.pdf", page_number=1, chunk_index=0),
            _doc(2, "Second chunk", file_name="a.pdf", page_number=2, chunk_index=1),
        ]
        result = build_context(docs)

        assert "[Source 1" in result.context_text
        assert "[Source 2" in result.context_text
        assert "First chunk" in result.context_text
        assert "Second chunk" in result.context_text
        # Sources are separated by double newline
        assert "\n\n" in result.context_text

    def test_source_numbers_are_sequential(self):
        docs = [_doc(i, f"chunk {i}") for i in range(5)]
        result = build_context(docs, max_chars=100_000)

        for i in range(1, 6):
            assert f"[Source {i}" in result.context_text


# ---------------------------------------------------------------------------
# build_context — source metadata
# ---------------------------------------------------------------------------

class TestBuildContextMetadata:
    def test_source_meta_preserves_all_fields(self):
        doc = _doc(42, "text", file_name="report.pdf", page_number=3,
                   chunk_index=7, title="Report (Page 3, Chunk 7)")
        result = build_context([doc])

        assert len(result.sources) == 1
        src = result.sources[0]
        assert isinstance(src, SourceMeta)
        assert src.id == 42
        assert src.title == "Report (Page 3, Chunk 7)"
        assert src.file_name == "report.pdf"
        assert src.page_number == 3
        assert src.chunk_index == 7
        assert "report.pdf" in src.label

    def test_sources_in_input_order(self):
        docs = [
            _doc(10, "A", file_name="x.pdf", page_number=1, chunk_index=0),
            _doc(20, "B", file_name="y.pdf", page_number=2, chunk_index=1),
            _doc(30, "C", file_name="z.pdf", page_number=3, chunk_index=2),
        ]
        result = build_context(docs, max_chars=100_000)

        ids = [s.id for s in result.sources]
        assert ids == [10, 20, 30]


# ---------------------------------------------------------------------------
# build_context — character budget
# ---------------------------------------------------------------------------

class TestBuildContextBudget:
    def test_all_chunks_fit_within_budget(self):
        docs = [_doc(i, "short") for i in range(3)]
        result = build_context(docs, max_chars=100_000)

        assert result.chunks_included == 3
        assert result.chunks_dropped == 0

    def test_budget_drops_excess_chunks(self):
        """With a tiny budget, only the first chunk should fit."""
        docs = [
            _doc(1, "A" * 50, file_name="a.pdf", page_number=1, chunk_index=0),
            _doc(2, "B" * 50, file_name="a.pdf", page_number=1, chunk_index=1),
            _doc(3, "C" * 50, file_name="a.pdf", page_number=2, chunk_index=2),
        ]
        # Budget just enough for the first chunk's block (label + content)
        first_block = build_context([docs[0]], max_chars=100_000)
        tight_budget = first_block.total_chars + 1  # barely fits one

        result = build_context(docs, max_chars=tight_budget)

        assert result.chunks_included == 1
        assert result.chunks_dropped == 2
        assert result.sources[0].id == 1

    def test_budget_too_small_for_any_chunk(self):
        """If the budget is smaller than the first chunk, include nothing."""
        doc = _doc(1, "A" * 200, file_name="big.pdf", page_number=1, chunk_index=0)
        result = build_context([doc], max_chars=10)

        assert result.chunks_included == 0
        assert result.chunks_dropped == 1
        assert result.context_text == ""

    def test_total_chars_matches_context_length(self):
        docs = [_doc(i, f"content number {i}", file_name="f.pdf") for i in range(4)]
        result = build_context(docs, max_chars=100_000)

        assert result.total_chars == len(result.context_text)

    def test_does_not_truncate_mid_chunk(self):
        """Chunks are either fully included or fully excluded — no partial text."""
        content = "This is a complete sentence that should not be cut."
        docs = [_doc(1, content, file_name="a.pdf", page_number=1, chunk_index=0)]
        result = build_context(docs, max_chars=100_000)

        assert content in result.context_text


# ---------------------------------------------------------------------------
# build_context — validation
# ---------------------------------------------------------------------------

class TestBuildContextValidation:
    def test_raises_on_zero_max_chars(self):
        with pytest.raises(ValueError, match="max_chars must be a positive integer"):
            build_context([], max_chars=0)

    def test_raises_on_negative_max_chars(self):
        with pytest.raises(ValueError, match="max_chars must be a positive integer"):
            build_context([], max_chars=-100)

    def test_raises_on_non_int_max_chars(self):
        with pytest.raises(ValueError, match="max_chars must be a positive integer"):
            build_context([], max_chars=12.5)
