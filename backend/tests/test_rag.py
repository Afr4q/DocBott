"""
Tests for the RAG engine.
Uses mock retrieval results - does not require actual embeddings.
"""

import pytest
from unittest.mock import patch, MagicMock
from app.rag import (
    RAGResponse,
    _student_answer,
    _teacher_answer,
    _researcher_answer,
    generate_question_suggestions,
)


class TestRAGResponse:
    """Tests for the RAGResponse dataclass."""

    def test_creation(self):
        r = RAGResponse(
            answer="Test answer",
            sources=[{"page": 1, "content": "source text"}],
            confidence=0.85,
            reasoning="Based on 3 sources",
            role="student",
            suggestions=["Follow-up question?"],
        )
        assert r.answer == "Test answer"
        assert r.confidence == 0.85
        assert r.role == "student"
        assert len(r.sources) == 1
        assert len(r.suggestions) == 1


class TestRoleBasedAnswers:
    """Tests for role-specific answer formatting."""

    def test_student_answer(self):
        chunks = [
            "Machine learning is a type of artificial intelligence.",
            "It allows computers to learn from data without being explicitly programmed.",
            "Common algorithms include decision trees and neural networks.",
        ]
        answer = _student_answer("What is machine learning?", chunks)
        assert len(answer) > 0
        assert isinstance(answer, str)

    def test_teacher_answer(self):
        chunks = [
            "Python was created by Guido van Rossum in 1991.",
            "It emphasizes code readability and simplicity.",
        ]
        answer = _teacher_answer("Tell me about Python", chunks)
        assert len(answer) > 0
        assert isinstance(answer, str)

    def test_researcher_answer(self):
        chunks = [
            "The experiment showed a 15% improvement in accuracy.",
            "Previous studies reported mixed results on this topic.",
        ]
        answer = _researcher_answer("What were the results?", chunks)
        assert len(answer) > 0
        assert isinstance(answer, str)

    def test_student_answer_empty_chunks(self):
        answer = _student_answer("What is AI?", [])
        assert isinstance(answer, str)

    def test_teacher_answer_single_chunk(self):
        answer = _teacher_answer("Explain gravity", ["Gravity is a fundamental force."])
        assert "Gravity" in answer or "gravity" in answer


class TestQuestionSuggestions:
    """Tests for question suggestion generation."""

    def test_generates_suggestions(self):
        text = "Machine learning uses algorithms to learn from data. Deep learning is a subset. Neural networks are used."
        suggestions = generate_question_suggestions(text)
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0

    def test_empty_text(self):
        suggestions = generate_question_suggestions("")
        assert isinstance(suggestions, list)
