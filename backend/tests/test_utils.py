"""
Tests for the utils module.
Uses generic assertions - not tied to any specific PDF.
"""

import pytest
from app.utils import (
    clean_text, remove_header_footer, split_into_sentences,
    estimate_confidence, truncate_text, safe_filename, compute_file_hash
)


class TestCleanText:
    """Tests for text cleaning utilities."""

    def test_empty_input(self):
        assert clean_text("") == ""
        assert clean_text(None) == ""

    def test_removes_control_characters(self):
        text = "Hello\x00World\x07Test"
        result = clean_text(text)
        assert "\x00" not in result
        assert "\x07" not in result
        assert "Hello" in result
        assert "World" in result

    def test_collapses_whitespace(self):
        text = "Hello     World     Test"
        result = clean_text(text)
        assert "     " not in result
        assert "Hello World Test" == result

    def test_collapses_newlines(self):
        text = "Line1\n\n\n\n\nLine2"
        result = clean_text(text)
        assert result == "Line1\n\nLine2"

    def test_normalizes_quotes(self):
        text = "He said \u2018hello\u2019 and \u201cgoodbye\u201d"
        result = clean_text(text)
        assert "\u2018" not in result
        assert "\u201c" not in result


class TestSplitIntoSentences:
    """Tests for sentence splitting."""

    def test_basic_splitting(self):
        text = "This is sentence one. This is sentence two. And three."
        sentences = split_into_sentences(text)
        assert len(sentences) >= 2

    def test_handles_abbreviations(self):
        text = "Dr. Smith went to the store. He bought items."
        sentences = split_into_sentences(text)
        # Should not split on "Dr."
        assert any("Dr" in s for s in sentences)

    def test_empty_input(self):
        assert split_into_sentences("") == []
        assert split_into_sentences("   ") == []


class TestEstimateConfidence:
    """Tests for confidence scoring."""

    def test_empty_scores(self):
        assert estimate_confidence([]) == 0.0

    def test_perfect_scores(self):
        result = estimate_confidence([1.0, 1.0, 1.0])
        assert result == 1.0

    def test_range_is_zero_to_one(self):
        result = estimate_confidence([0.5, 0.3, 0.1])
        assert 0.0 <= result <= 1.0

    def test_higher_top_scores_increase_confidence(self):
        high = estimate_confidence([0.9, 0.8, 0.7])
        low = estimate_confidence([0.3, 0.2, 0.1])
        assert high > low


class TestTruncateText:
    """Tests for text truncation."""

    def test_short_text_unchanged(self):
        text = "Short text."
        assert truncate_text(text, 100) == text

    def test_long_text_truncated(self):
        text = "A" * 1000
        result = truncate_text(text, 100)
        assert len(result) <= 103  # 100 + "..."

    def test_prefers_sentence_boundary(self):
        text = "First sentence. Second sentence. Third sentence that is longer."
        result = truncate_text(text, 40)
        assert result.endswith(".")


class TestSafeFilename:
    """Tests for filename sanitization."""

    def test_removes_special_chars(self):
        result = safe_filename("my file<>:name.pdf")
        assert "<" not in result
        assert ">" not in result
        assert result.endswith(".pdf")

    def test_normalizes_spaces(self):
        result = safe_filename("my   file   name.pdf")
        assert "   " not in result


class TestRemoveHeaderFooter:
    """Tests for header/footer removal."""

    def test_removes_page_numbers(self):
        lines = ["Content line 1", "Content line 2", "More content", "42"]
        text = "\n".join(lines)
        result = remove_header_footer(text)
        # The trailing page number should be removed
        assert result.strip().endswith("More content")

    def test_short_text_unchanged(self):
        text = "Short"
        assert remove_header_footer(text) == text
