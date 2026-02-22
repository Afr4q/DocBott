"""
RAG Engine - Retrieval-Augmented Generation.
Combines retrieved document chunks with role-based answer generation.
NEVER hallucinates - answers are strictly grounded in retrieved content.
Supports multiple answer roles: student, teacher, researcher.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field

from app.retrieval import hybrid_search, RetrievalResult, rerank_results
from app.utils import (
    estimate_confidence, truncate_text, split_into_sentences,
    clean_chunk_for_display, get_logger
)
from app.database import UserRole

logger = get_logger(__name__)


@dataclass
class RAGResponse:
    """Complete RAG response with answer, sources, and metadata."""
    answer: str = ""
    ai_summary: str = ""
    confidence: float = 0.0
    sources: List[Dict] = field(default_factory=list)
    reasoning: str = ""
    role: str = "student"
    query: str = ""
    suggested_questions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "answer": self.answer,
            "ai_summary": self.ai_summary,
            "confidence": self.confidence,
            "sources": self.sources,
            "reasoning": self.reasoning,
            "role": self.role,
            "query": self.query,
            "suggested_questions": self.suggested_questions,
        }


# ──────────────────────────────────────────────
# Answer Generation (Extractive, role-based)
# ──────────────────────────────────────────────
def generate_extractive_answer(
    query: str,
    results: List[RetrievalResult],
    role: str = "student",
) -> str:
    """
    Generate an extractive answer directly from retrieved content.
    This is the grounded answer - no hallucination possible.

    Role-based formatting:
    - student: Short, simple answer from best-matching chunk.
    - teacher: Structured answer with key points.
    - researcher: Full answer with citations.
    """
    if not results:
        return "No relevant information found in the uploaded documents."

    if role == "student":
        return _student_answer(query, results)
    elif role == "teacher":
        return _teacher_answer(query, results)
    else:
        return _researcher_answer(query, results)


def _clean_result_content(content: str) -> str:
    """Clean a chunk's content before using it in an answer."""
    return clean_chunk_for_display(content)


def _student_answer(query: str, results: List[RetrievalResult]) -> str:
    """Simple, concise answer from the best-matching content."""
    best = results[0]
    sentences = split_into_sentences(_clean_result_content(best.content))

    # Find most relevant sentences
    query_words = set(query.lower().split())
    scored = []
    for sent in sentences:
        overlap = len(query_words & set(sent.lower().split()))
        scored.append((sent, overlap))

    scored.sort(key=lambda x: x[1], reverse=True)

    # Return top 2-3 most relevant sentences
    top_sentences = [s[0] for s in scored[:3]]
    return " ".join(top_sentences) if top_sentences else truncate_text(best.content, 300)


def _teacher_answer(query: str, results: List[RetrievalResult]) -> str:
    """Structured answer with key points from multiple sources."""
    sections = []
    sections.append("**Key Points:**\n")

    for i, result in enumerate(results[:3]):
        sentences = split_into_sentences(_clean_result_content(result.content))
        # Pick most relevant sentences
        query_words = set(query.lower().split())
        relevant = [
            s for s in sentences
            if len(query_words & set(s.lower().split())) > 0
        ][:2]

        if relevant:
            point = " ".join(relevant)
            source = f"(Page {result.page_number})"
            sections.append(f"{i+1}. {point} {source}")

    if len(sections) == 1:
        # No specific points found, use general content
        return truncate_text(results[0].content, 500)

    return "\n".join(sections)


def _researcher_answer(query: str, results: List[RetrievalResult]) -> str:
    """Detailed answer with inline citations."""
    sections = []
    citation_index = 1

    for result in results[:4]:
        sentences = split_into_sentences(_clean_result_content(result.content))
        query_words = set(query.lower().split())

        relevant = [
            s for s in sentences
            if len(query_words & set(s.lower().split())) > 0
        ]

        for sent in relevant[:2]:
            citation = f"[{citation_index}]"
            sections.append(f"{sent} {citation}")
            citation_index += 1

    if not sections:
        sections.append(truncate_text(results[0].content, 600))

    # Add references section
    refs = ["\n\n**References:**"]
    for i, r in enumerate(results[:4]):
        refs.append(f"[{i+1}] {r.filename or 'Document'}, Page {r.page_number}")

    return "\n".join(sections + refs)


# ──────────────────────────────────────────────
# Question Suggestions
# ──────────────────────────────────────────────
def generate_question_suggestions(results: List[RetrievalResult]) -> List[str]:
    """
    Generate suggested follow-up questions based on document content.
    Extracts key topics and phrases to form relevant questions.
    """
    suggestions = []

    # Collect unique topics from results
    all_text = " ".join(r.content for r in results[:3])
    sentences = split_into_sentences(all_text)

    # Find sentences with key information markers
    for sent in sentences[:10]:
        lower = sent.lower()
        if any(marker in lower for marker in [
            "important", "significant", "key", "main",
            "result", "conclusion", "finding", "method",
            "defined as", "refers to", "means",
        ]):
            # Transform statement into a question
            if len(sent) > 20 and len(sent) < 200:
                # Simple heuristic to generate questions
                if "is" in lower:
                    suggestions.append(f"What {sent.split('is')[0].strip().lower()}?")
                elif "are" in lower:
                    suggestions.append(f"What {sent.split('are')[0].strip().lower()}?")
                else:
                    suggestions.append(f"Can you explain: {truncate_text(sent, 80)}?")

    return suggestions[:5]  # Return max 5 suggestions


# ──────────────────────────────────────────────
# Main RAG Query
# ──────────────────────────────────────────────
def query_rag(
    query: str,
    role: str = "student",
    document_ids: Optional[List[int]] = None,
    top_k: int = 5,
    enable_rerank: bool = False,
    filenames: Optional[Dict[int, str]] = None,
) -> RAGResponse:
    """
    Main RAG query entry point.

    1. Retrieves relevant chunks via hybrid search
    2. Optionally re-ranks results
    3. Generates extractive answer based on role
    4. Computes confidence score
    5. Generates question suggestions

    Args:
        query: User's question.
        role: User role for answer formatting.
        document_ids: Optional filter to specific documents.
        top_k: Number of chunks to retrieve.
        enable_rerank: Whether to use cross-encoder re-ranking.
        filenames: Mapping of document_id to filename for sources.

    Returns:
        RAGResponse with answer, sources, confidence, etc.
    """
    response = RAGResponse(query=query, role=role)
    filenames = filenames or {}

    try:
        # Step 1: Retrieve
        logger.info(f"RAG query: '{query}' (role={role})")
        results = hybrid_search(query, document_ids=document_ids, top_k=top_k)

        if not results:
            response.answer = "No relevant information found in the uploaded documents."
            response.reasoning = "No matching content retrieved from document store."
            return response

        # Step 2: Optional re-ranking
        if enable_rerank:
            results = rerank_results(query, results, top_k=top_k)

        # Enrich with filenames
        for r in results:
            r.filename = filenames.get(r.document_id, f"Document {r.document_id}")

        # Step 3: Generate extractive answer
        response.answer = generate_extractive_answer(query, results, role)

        # Step 4: Build sources
        response.sources = [
            {
                "file": r.filename,
                "page": r.page_number,
                "score": r.final_score,
                "source_type": r.source_type,
                "content_preview": truncate_text(r.content, 150),
            }
            for r in results
        ]

        # Step 5: Confidence scoring
        scores = [r.final_score for r in results]
        response.confidence = estimate_confidence(scores)

        # Step 6: Reasoning
        response.reasoning = (
            f"Answer derived from {len(results)} document segments "
            f"across {len(set(r.document_id for r in results))} document(s). "
            f"Top score: {max(scores):.4f}, "
            f"Average score: {sum(scores)/len(scores):.4f}."
        )

        # Step 7: Question suggestions
        response.suggested_questions = generate_question_suggestions(results)

    except Exception as e:
        logger.error(f"RAG query failed: {e}")
        response.answer = "An error occurred while processing your question."
        response.reasoning = f"Error: {str(e)}"

    return response


def compare_across_documents(
    query: str,
    document_ids: List[int],
    role: str = "student",
    filenames: Optional[Dict[int, str]] = None,
) -> Dict:
    """
    Compare answers across multiple documents.
    Runs RAG query filtered to each document individually.

    Returns:
        Dict with per-document answers and a comparison summary.
    """
    filenames = filenames or {}
    comparisons = []

    for doc_id in document_ids:
        result = query_rag(
            query=query,
            role=role,
            document_ids=[doc_id],
            filenames=filenames,
        )
        comparisons.append({
            "document_id": doc_id,
            "filename": filenames.get(doc_id, f"Document {doc_id}"),
            "answer": result.answer,
            "confidence": result.confidence,
            "sources": result.sources,
        })

    return {
        "query": query,
        "comparisons": comparisons,
        "document_count": len(document_ids),
    }
