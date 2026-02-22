"""
Tests for the retrieval module.
Uses mock data - does not require a real index or embeddings.
"""

import pytest
from app.retrieval import RetrievalResult, normalize_scores, BM25Index


class TestRetrievalResult:
    """Tests for the RetrievalResult dataclass."""

    def test_creation(self):
        r = RetrievalResult(
            content="Test content",
            page_number=3,
            document_id=1,
            filename="test.pdf",
            vector_score=0.85,
            bm25_score=0.6,
            combined_score=0.78,
        )
        assert r.content == "Test content"
        assert r.page_number == 3
        assert r.combined_score == 0.78

    def test_defaults(self):
        r = RetrievalResult(
            content="test",
            page_number=1,
            document_id=1,
            filename="f.pdf",
        )
        assert r.vector_score == 0.0
        assert r.bm25_score == 0.0
        assert r.combined_score == 0.0


class TestNormalizeScores:
    """Tests for score normalization."""

    def test_normalize_basic(self):
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = normalize_scores(scores)
        assert len(result) == 5
        assert min(result) == 0.0
        assert max(result) == 1.0

    def test_normalize_single_value(self):
        scores = [5.0]
        result = normalize_scores(scores)
        assert result == [1.0]

    def test_normalize_identical_values(self):
        scores = [3.0, 3.0, 3.0]
        result = normalize_scores(scores)
        # All same -> all should be same (typically 0 or 1)
        assert len(set(result)) <= 1

    def test_normalize_empty(self):
        scores = []
        result = normalize_scores(scores)
        assert result == []

    def test_normalize_preserves_order(self):
        scores = [10.0, 5.0, 20.0, 1.0]
        result = normalize_scores(scores)
        # 20 should have highest normalized score, 1 should have lowest
        assert result[2] == max(result)
        assert result[3] == min(result)


class TestBM25Index:
    """Tests for the BM25 keyword index."""

    def test_build_and_search(self):
        documents = [
            "machine learning algorithms for classification",
            "deep neural networks and natural language processing",
            "database systems and SQL queries",
            "machine learning for natural language understanding",
        ]
        idx = BM25Index()
        idx.build(documents)

        results = idx.search("machine learning", top_k=2)
        assert len(results) <= 2
        # Results should be tuples of (index, score)
        for doc_idx, score in results:
            assert 0 <= doc_idx < len(documents)
            assert score >= 0

    def test_search_no_match(self):
        documents = ["cat sat on mat", "dog ran in park"]
        idx = BM25Index()
        idx.build(documents)

        results = idx.search("quantum physics relativity", top_k=2)
        # May return results with very low scores or empty
        assert isinstance(results, list)

    def test_empty_index(self):
        idx = BM25Index()
        idx.build([])
        results = idx.search("anything", top_k=5)
        assert results == []
