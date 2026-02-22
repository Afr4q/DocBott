"""
Tests for the evaluation module.
Uses synthetic answers and references.
"""

import pytest
from app.evaluation import evaluate_answer, compute_bleu, compute_rouge


class TestEvaluateAnswer:
    """Tests for the evaluate_answer function."""

    def test_basic_evaluation(self):
        result = evaluate_answer(
            query="What is machine learning?",
            answer="Machine learning is a subset of AI that learns from data.",
            sources=["Machine learning uses data to learn patterns."],
        )
        assert "source_overlap" in result
        assert "query_relevance" in result
        assert "length_ratio" in result
        assert "groundedness" in result

    def test_high_overlap_answer(self):
        source = "Python is a popular programming language."
        result = evaluate_answer(
            query="What is Python?",
            answer=source,  # Exact match with source
            sources=[source],
        )
        assert result["source_overlap"] > 0.5

    def test_empty_sources(self):
        result = evaluate_answer(
            query="test query",
            answer="test answer",
            sources=[],
        )
        assert result["source_overlap"] == 0.0

    def test_empty_answer(self):
        result = evaluate_answer(
            query="What is AI?",
            answer="",
            sources=["AI is artificial intelligence."],
        )
        assert result["length_ratio"] == 0.0


class TestComputeBleu:
    """Tests for BLEU score computation."""

    def test_identical_strings(self):
        score = compute_bleu("the cat sat on the mat", "the cat sat on the mat")
        assert score > 0.9

    def test_different_strings(self):
        score = compute_bleu("the cat sat on the mat", "a dog ran in the park")
        assert score < 0.5

    def test_empty_strings(self):
        score = compute_bleu("", "")
        assert score == 0.0

    def test_partial_match(self):
        score = compute_bleu(
            "the quick brown fox jumps",
            "the quick brown fox leaps over",
        )
        assert 0.0 < score < 1.0


class TestComputeRouge:
    """Tests for ROUGE score computation."""

    def test_identical_strings(self):
        score = compute_rouge("the cat sat on the mat", "the cat sat on the mat")
        assert score > 0.9

    def test_different_strings(self):
        score = compute_rouge("the cat sat on the mat", "a dog ran in the park")
        assert score < 0.5

    def test_empty_strings(self):
        score = compute_rouge("", "")
        assert score == 0.0
