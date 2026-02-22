"""
Retrieval module - Hybrid search combining BM25 keyword search
and vector semantic search with score normalization and weighted fusion.
Optional cross-encoder re-ranking for improved precision.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from app.config import TOP_K_RESULTS, BM25_WEIGHT, VECTOR_WEIGHT
from app.indexing import search_vectors, generate_embeddings
from app.utils import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """A single retrieval result with source information."""
    content: str
    page_number: int
    document_id: int
    filename: str = ""
    source_type: str = "text"
    bm25_score: float = 0.0
    vector_score: float = 0.0
    final_score: float = 0.0
    chunk_id: str = ""


# ──────────────────────────────────────────────
# BM25 Index (in-memory, rebuilt per query scope)
# ──────────────────────────────────────────────
class BM25Index:
    """BM25 keyword search index using rank_bm25."""

    def __init__(self):
        self.corpus = []  # List of tokenized documents
        self.documents = []  # Original document dicts
        self.bm25 = None

    def build(self, documents: List[Dict]) -> None:
        """
        Build BM25 index from a list of document dicts.
        Each dict should have 'content' and 'metadata' keys.
        """
        from rank_bm25 import BM25Okapi

        self.documents = documents
        self.corpus = [self._tokenize(doc["content"]) for doc in documents]
        self.bm25 = BM25Okapi(self.corpus)
        logger.debug(f"BM25 index built with {len(self.corpus)} documents")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Search the BM25 index.
        Returns list of (document_index, score) tuples.
        """
        if not self.bm25:
            return []

        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices sorted by score
        indexed_scores = [(i, float(score)) for i, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        return indexed_scores[:top_k]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace tokenization with lowercasing."""
        return text.lower().split()


# ──────────────────────────────────────────────
# Score Normalization
# ──────────────────────────────────────────────
def normalize_scores(scores: List[float]) -> List[float]:
    """
    Min-max normalize scores to [0, 1] range.
    Handles edge cases of empty or uniform scores.
    """
    if not scores:
        return []

    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    return [(s - min_score) / (max_score - min_score) for s in scores]


# ──────────────────────────────────────────────
# Hybrid Search
# ──────────────────────────────────────────────
def hybrid_search(
    query: str,
    document_ids: Optional[List[int]] = None,
    top_k: int = None,
    bm25_weight: float = None,
    vector_weight: float = None,
) -> List[RetrievalResult]:
    """
    Perform hybrid search combining BM25 and vector search.

    Steps:
    1. Vector search (semantic similarity)
    2. BM25 keyword search (over vector results corpus)
    3. Score normalization
    4. Weighted fusion
    5. Optional re-ranking

    Args:
        query: The search query.
        document_ids: Optional filter to specific documents.
        top_k: Number of results to return.
        bm25_weight: Weight for BM25 scores.
        vector_weight: Weight for vector scores.

    Returns:
        List of RetrievalResult sorted by final_score.
    """
    top_k = top_k or TOP_K_RESULTS
    bm25_weight = bm25_weight or BM25_WEIGHT
    vector_weight = vector_weight or VECTOR_WEIGHT

    # Step 1: Vector search (get more than needed for fusion)
    search_k = top_k * 3
    filter_dict = None
    if document_ids and len(document_ids) == 1:
        filter_dict = {"document_id": document_ids[0]}

    vector_results = search_vectors(query, top_k=search_k, filter_dict=filter_dict)

    if not vector_results:
        logger.warning("No vector results found")
        return []

    # Filter by document_ids if multiple specified
    if document_ids and len(document_ids) > 1:
        vector_results = [
            r for r in vector_results
            if r.get("metadata", {}).get("document_id") in document_ids
        ]

    # Step 2: Build BM25 index from vector results corpus
    bm25_index = BM25Index()
    bm25_docs = [
        {"content": r["content"], "metadata": r.get("metadata", {})}
        for r in vector_results
    ]
    bm25_index.build(bm25_docs)
    bm25_results = bm25_index.search(query, top_k=len(vector_results))

    # Step 3: Create unified score map
    score_map = {}  # index -> {bm25_score, vector_score}

    for i, vr in enumerate(vector_results):
        score_map[i] = {
            "content": vr["content"],
            "metadata": vr.get("metadata", {}),
            "vector_score": vr.get("score", 0),
            "bm25_score": 0.0,
        }

    for idx, bm25_score in bm25_results:
        if idx in score_map:
            score_map[idx]["bm25_score"] = bm25_score

    # Step 4: Normalize and fuse scores
    vector_scores = [sm["vector_score"] for sm in score_map.values()]
    bm25_scores = [sm["bm25_score"] for sm in score_map.values()]

    norm_vector = normalize_scores(vector_scores)
    norm_bm25 = normalize_scores(bm25_scores)

    results = []
    for i, (idx, sm) in enumerate(score_map.items()):
        # Weighted fusion
        final_score = (
            vector_weight * norm_vector[i] +
            bm25_weight * norm_bm25[i]
        )

        meta = sm["metadata"]
        results.append(RetrievalResult(
            content=sm["content"],
            page_number=meta.get("page_number", 0),
            document_id=meta.get("document_id", 0),
            source_type=meta.get("source_type", "text"),
            bm25_score=norm_bm25[i],
            vector_score=norm_vector[i],
            final_score=round(final_score, 4),
        ))

    # Sort by final_score descending
    results.sort(key=lambda r: r.final_score, reverse=True)

    # Return top_k
    return results[:top_k]


def rerank_results(
    query: str,
    results: List[RetrievalResult],
    top_k: int = 5,
) -> List[RetrievalResult]:
    """
    Optional cross-encoder re-ranking for improved precision.
    Uses a lightweight cross-encoder model.
    """
    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # Create query-document pairs
        pairs = [(query, r.content) for r in results]
        scores = model.predict(pairs)

        # Update scores
        for i, score in enumerate(scores):
            results[i].final_score = float(score)

        results.sort(key=lambda r: r.final_score, reverse=True)
        logger.info("Re-ranking complete")

    except ImportError:
        logger.warning("Cross-encoder not available, skipping re-ranking")
    except Exception as e:
        logger.error(f"Re-ranking failed: {e}")

    return results[:top_k]
