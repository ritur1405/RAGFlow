"""Tests for backend/app/utils.py — Week 2 functions.

Covers:
  - extract_text_from_pdf  (mocked PdfReader + corrupt-PDF guard)
  - clean_text             (direct string input)
  - chunk_pages            (direct dict input)
  - Real-PDF integration   (end-to-end with pypdf.PdfWriter)
"""

import io

import pytest
from unittest.mock import MagicMock, patch
from pypdf import PdfWriter
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from app.utils import clean_text, chunk_pages, extract_text_from_pdf


# ── extract_text_from_pdf ─────────────────────────────────────────────────


class TestExtractTextFromPdf:
    """Tests for PDF text extraction (PdfReader is mocked)."""

    def _make_mock_page(self, text: str) -> MagicMock:
        page = MagicMock()
        page.extract_text.return_value = text
        return page

    @patch("app.utils.PdfReader")
    def test_single_page_pdf(self, mock_reader_cls):
        mock_reader = MagicMock()
        mock_reader.pages = [self._make_mock_page("Hello world")]
        mock_reader_cls.return_value = mock_reader

        result = extract_text_from_pdf(b"fake-pdf-bytes")

        assert len(result) == 1
        assert result[0]["page_number"] == 1
        assert result[0]["text"] == "Hello world"

    @patch("app.utils.PdfReader")
    def test_multi_page_pdf(self, mock_reader_cls):
        mock_reader = MagicMock()
        mock_reader.pages = [
            self._make_mock_page("Page one text"),
            self._make_mock_page("Page two text"),
            self._make_mock_page("Page three text"),
        ]
        mock_reader_cls.return_value = mock_reader

        result = extract_text_from_pdf(b"fake-pdf-bytes")

        assert len(result) == 3
        assert result[0]["page_number"] == 1
        assert result[1]["page_number"] == 2
        assert result[2]["page_number"] == 3

    @patch("app.utils.PdfReader")
    def test_skips_blank_pages(self, mock_reader_cls):
        mock_reader = MagicMock()
        mock_reader.pages = [
            self._make_mock_page("Page one text"),
            self._make_mock_page("   "),  # blank
            self._make_mock_page(""),  # empty
            self._make_mock_page("Page four text"),
        ]
        mock_reader_cls.return_value = mock_reader

        result = extract_text_from_pdf(b"fake-pdf-bytes")

        assert len(result) == 2
        assert result[0]["page_number"] == 1
        assert result[1]["page_number"] == 4  # original page index preserved

    @patch("app.utils.PdfReader")
    def test_raises_on_no_pages(self, mock_reader_cls):
        mock_reader = MagicMock()
        mock_reader.pages = []
        mock_reader_cls.return_value = mock_reader

        with pytest.raises(ValueError, match="no pages"):
            extract_text_from_pdf(b"fake-pdf-bytes")

    @patch("app.utils.PdfReader")
    def test_raises_on_all_blank_pages(self, mock_reader_cls):
        mock_reader = MagicMock()
        mock_reader.pages = [
            self._make_mock_page(""),
            self._make_mock_page("   \n  "),
        ]
        mock_reader_cls.return_value = mock_reader

        with pytest.raises(ValueError, match="no extractable text"):
            extract_text_from_pdf(b"fake-pdf-bytes")

    @patch("app.utils.PdfReader")
    def test_handles_none_from_extract_text(self, mock_reader_cls):
        """Some PDF pages return None instead of empty string."""
        page = MagicMock()
        page.extract_text.return_value = None
        mock_reader = MagicMock()
        mock_reader.pages = [page, self._make_mock_page("Real text")]
        mock_reader_cls.return_value = mock_reader

        result = extract_text_from_pdf(b"fake-pdf-bytes")

        assert len(result) == 1
        assert result[0]["page_number"] == 2

    def test_raises_on_corrupt_bytes(self):
        """Corrupt or non-PDF bytes should raise a clear ValueError."""
        with pytest.raises(ValueError, match="Invalid or corrupt PDF file"):
            extract_text_from_pdf(b"this is not a pdf")

    def test_raises_on_empty_bytes(self):
        """Empty bytes should raise a clear ValueError."""
        with pytest.raises(ValueError, match="Invalid or corrupt PDF file"):
            extract_text_from_pdf(b"")


# ── clean_text ────────────────────────────────────────────────────────────


class TestCleanText:
    """Tests for text cleaning."""

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_none_returns_empty(self):
        assert clean_text(None) == ""

    def test_rejoins_hyphenated_line_breaks(self):
        assert clean_text("knowl-\nedge") == "knowledge"
        assert clean_text("multi-\nline") == "multiline"

    def test_replaces_soft_newlines_with_spaces(self):
        result = clean_text("hello\nworld")
        assert result == "hello world"

    def test_preserves_paragraph_breaks(self):
        result = clean_text("paragraph one\n\nparagraph two")
        assert result == "paragraph one\n\nparagraph two"

    def test_collapses_excessive_newlines(self):
        result = clean_text("a\n\n\n\n\nb")
        assert result == "a\n\nb"

    def test_collapses_spaces_and_tabs(self):
        result = clean_text("hello   \t\t  world")
        assert result == "hello world"

    def test_strips_leading_trailing_whitespace(self):
        result = clean_text("   hello world   ")
        assert result == "hello world"

    def test_combined_cleaning(self):
        raw = "  This is a  sen-\ntence with   extra\n\n\n\nspaces.  "
        result = clean_text(raw)
        assert result == "This is a sentence with extra\n\nspaces."


# ── chunk_pages ───────────────────────────────────────────────────────────


class TestChunkPages:
    """Tests for page-aware chunking with metadata preservation."""

    def test_single_page_single_chunk(self):
        pages = [{"page_number": 1, "text": "short text"}]
        result = chunk_pages(pages, chunk_size=500, chunk_overlap=50)

        assert len(result) == 1
        assert result[0]["content"] == "short text"
        assert result[0]["page_number"] == 1
        assert result[0]["chunk_index"] == 0

    def test_single_page_multiple_chunks(self):
        pages = [{"page_number": 1, "text": "a" * 120}]
        result = chunk_pages(pages, chunk_size=50, chunk_overlap=10)

        assert len(result) == 3  # 120 chars, step=40 → offsets 0,40,80
        for i, chunk in enumerate(result):
            assert chunk["page_number"] == 1
            assert chunk["chunk_index"] == i

    def test_multiple_pages_global_chunk_index(self):
        pages = [
            {"page_number": 1, "text": "a" * 100},
            {"page_number": 2, "text": "b" * 100},
        ]
        result = chunk_pages(pages, chunk_size=50, chunk_overlap=10)

        # Each page produces 3 chunks → 6 total
        assert len(result) == 6
        # Global chunk_index should be sequential
        for i, chunk in enumerate(result):
            assert chunk["chunk_index"] == i
        # First 3 from page 1, last 3 from page 2
        assert all(c["page_number"] == 1 for c in result[:3])
        assert all(c["page_number"] == 2 for c in result[3:])

    def test_preserves_original_page_numbers(self):
        """Page numbers from extraction (e.g. page 3) must be preserved."""
        pages = [{"page_number": 3, "text": "content from page three"}]
        result = chunk_pages(pages, chunk_size=500)

        assert result[0]["page_number"] == 3

    def test_empty_pages_list(self):
        result = chunk_pages([], chunk_size=500, chunk_overlap=50)
        assert result == []

    def test_overlap_larger_than_size_raises(self):
        pages = [{"page_number": 1, "text": "hello"}]
        with pytest.raises(ValueError, match="chunk_overlap must be less"):
            chunk_pages(pages, chunk_size=50, chunk_overlap=50)

    def test_chunk_content_is_stripped(self):
        pages = [{"page_number": 1, "text": "  hello  "}]
        result = chunk_pages(pages, chunk_size=500)

        assert result[0]["content"] == "hello"

    def test_output_dict_keys(self):
        pages = [{"page_number": 1, "text": "some content"}]
        result = chunk_pages(pages, chunk_size=500)

        assert set(result[0].keys()) == {"content", "page_number", "chunk_index"}


# ── Real-PDF Integration Test ─────────────────────────────────────────────


def _create_test_pdf(page_texts: list[str]) -> bytes:
    """Creates a real PDF in memory using reportlab with the given page texts."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    for text in page_texts:
        # Write text near the top of the page
        y = 750
        for line in text.split("\n"):
            c.drawString(72, y, line)
            y -= 14
        c.showPage()
    c.save()
    return buf.getvalue()


class TestRealPdfIntegration:
    """End-to-end test using a real PDF generated by reportlab."""

    def test_full_pipeline_with_real_pdf(self):
        """Create a real multi-page PDF, extract → clean → chunk, and validate output."""
        page1_text = "Machine learning is a subset of artificial intelligence."
        page2_text = "Natural language processing enables computers to understand text."

        pdf_bytes = _create_test_pdf([page1_text, page2_text])

        # 1. Extract
        pages = extract_text_from_pdf(pdf_bytes)
        assert len(pages) == 2
        assert pages[0]["page_number"] == 1
        assert pages[1]["page_number"] == 2

        # 2. Clean
        cleaned_pages = []
        for page in pages:
            cleaned = clean_text(page["text"])
            if cleaned:
                cleaned_pages.append(
                    {"page_number": page["page_number"], "text": cleaned}
                )
        assert len(cleaned_pages) == 2

        # 3. Chunk
        chunks = chunk_pages(cleaned_pages, chunk_size=500, chunk_overlap=50)
        assert len(chunks) >= 2  # at least one chunk per page

        # Validate structure of every chunk
        for i, chunk in enumerate(chunks):
            assert set(chunk.keys()) == {"content", "page_number", "chunk_index"}
            assert chunk["chunk_index"] == i
            assert isinstance(chunk["content"], str)
            assert len(chunk["content"]) > 0
            assert chunk["page_number"] in (1, 2)

    def test_single_page_real_pdf(self):
        """A single-page PDF should produce at least one chunk."""
        pdf_bytes = _create_test_pdf(["Hello world from a real PDF."])

        pages = extract_text_from_pdf(pdf_bytes)
        assert len(pages) == 1
        assert "Hello" in pages[0]["text"] or "world" in pages[0]["text"]

        chunks = chunk_pages(
            [{"page_number": p["page_number"], "text": clean_text(p["text"])}
             for p in pages],
            chunk_size=500,
            chunk_overlap=50,
        )
        assert len(chunks) >= 1
        assert chunks[0]["page_number"] == 1
        assert chunks[0]["chunk_index"] == 0
