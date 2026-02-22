"""
Tests for the chunking module.
Uses generic text - not tied to any specific document.
"""

import pytest
from app.chunking import chunk_text, chunk_document, Chunk


class TestChunkText:
    """Tests for text chunking."""

    def test_empty_text(self):
        result = chunk_text("", document_id=1, page_number=1)
        assert result == []

    def test_short_text_single_chunk(self):
        text = "This is a short sentence. It fits in one chunk."
        result = chunk_text(text, document_id=1, page_number=1, chunk_size=500)
        assert len(result) == 1
        assert result[0].content == text

    def test_long_text_multiple_chunks(self):
        # Generate text that exceeds chunk_size
        sentences = [f"This is sentence number {i}." for i in range(100)]
        text = " ".join(sentences)
        result = chunk_text(text, document_id=1, page_number=1, chunk_size=200)
        assert len(result) > 1

    def test_chunk_metadata(self):
        text = "Test sentence for metadata verification."
        result = chunk_text(text, document_id=5, page_number=3, source_type="ocr")
        assert len(result) == 1
        chunk = result[0]
        assert chunk.document_id == 5
        assert chunk.page_number == 3
        assert chunk.source_type == "ocr"
        assert chunk.char_count > 0

    def test_chunk_overlap(self):
        sentences = [f"Sentence {i} with some content here." for i in range(50)]
        text = " ".join(sentences)
        result = chunk_text(text, document_id=1, page_number=1, chunk_size=200, chunk_overlap=50)

        if len(result) >= 2:
            # Check that chunks have some overlapping content
            first_words = set(result[0].content.split())
            second_words = set(result[1].content.split())
            overlap = first_words & second_words
            assert len(overlap) > 0, "Chunks should have overlapping content"


class TestChunkDocument:
    """Tests for document-level chunking."""

    def test_empty_pages(self):
        result = chunk_document([], document_id=1)
        assert result == []

    def test_multiple_pages(self):
        pages = [
            {"page_number": 1, "merged_text": "Content on page one.", "source_types": ["text"]},
            {"page_number": 2, "merged_text": "Content on page two.", "source_types": ["text"]},
        ]
        result = chunk_document(pages, document_id=1)
        assert len(result) >= 2

    def test_skips_empty_pages(self):
        pages = [
            {"page_number": 1, "merged_text": "Has content.", "source_types": ["text"]},
            {"page_number": 2, "merged_text": "", "source_types": ["text"]},
            {"page_number": 3, "merged_text": "Also has content.", "source_types": ["text"]},
        ]
        result = chunk_document(pages, document_id=1)
        page_numbers = {c.page_number for c in result}
        assert 2 not in page_numbers

    def test_sequential_indexing(self):
        pages = [
            {"page_number": 1, "merged_text": "Content one.", "source_types": ["text"]},
            {"page_number": 2, "merged_text": "Content two.", "source_types": ["text"]},
        ]
        result = chunk_document(pages, document_id=1)
        indices = [c.chunk_index for c in result]
        assert indices == list(range(len(indices)))
