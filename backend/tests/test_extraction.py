"""
Tests for the extraction module.
Uses mock file objects - does not require actual PDF files.
"""

import pytest
from unittest.mock import MagicMock, patch
from app.extraction import PageContent, ExtractionResult


class TestPageContent:
    """Tests for PageContent dataclass."""

    def test_page_content_creation(self):
        pc = PageContent(
            page_number=1,
            text="Hello world",
            has_images=False,
            has_tables=False,
            char_count=11,
        )
        assert pc.page_number == 1
        assert pc.text == "Hello world"
        assert pc.char_count == 11
        assert pc.has_images is False
        assert pc.has_tables is False

    def test_page_content_defaults(self):
        pc = PageContent(page_number=1, text="test")
        assert pc.has_images is False
        assert pc.has_tables is False
        assert pc.char_count == 0


class TestExtractionResult:
    """Tests for ExtractionResult dataclass."""

    def test_extraction_result_creation(self):
        pages = [
            PageContent(page_number=1, text="Page 1"),
            PageContent(page_number=2, text="Page 2"),
        ]
        result = ExtractionResult(
            pages=pages,
            total_pages=2,
            metadata={"title": "Test"},
        )
        assert result.total_pages == 2
        assert len(result.pages) == 2
        assert result.metadata["title"] == "Test"

    def test_extraction_result_defaults(self):
        result = ExtractionResult(pages=[], total_pages=0)
        assert result.metadata == {}


class TestExtractMetadata:
    """Tests for the extract_metadata function using mocked PDF objects."""

    @patch("app.extraction.fitz")
    def test_extract_metadata(self, mock_fitz):
        mock_doc = MagicMock()
        mock_doc.metadata = {
            "title": "Test Doc",
            "author": "Author",
            "subject": "Subject",
            "creator": "Creator",
        }
        mock_doc.page_count = 5
        mock_fitz.open.return_value.__enter__ = MagicMock(return_value=mock_doc)
        mock_fitz.open.return_value.__exit__ = MagicMock(return_value=False)

        from app.extraction import extract_metadata
        result = extract_metadata("fake.pdf")
        assert result["title"] == "Test Doc"
        assert result["page_count"] == 5


class TestPageAnalysis:
    """Tests for page analysis helper functions."""

    def test_page_has_tables_heuristic(self):
        """Test table detection heuristic on text patterns."""
        from app.extraction import page_has_tables

        # Text with table-like patterns
        table_text = "Column1 | Column2 | Column3\nvalue1 | value2 | value3"
        assert page_has_tables(table_text) is True

        # Plain text without tables
        plain_text = "This is a simple paragraph without any table structures."
        assert page_has_tables(plain_text) is False
