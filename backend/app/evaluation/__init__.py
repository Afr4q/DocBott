"""
Evaluation module - Answer quality metrics (BLEU, ROUGE, overlap).
Provides quantitative assessment of RAG answer quality.
"""

from typing import Dict, List, Optional

from app.utils import get_logger

logger = get_logger(__name__)


def compute_bleu(reference: str, hypothesis: str) -> float:
    """
    Compute BLEU score between reference and hypothesis texts.
    Uses simple unigram/bigram BLEU for efficiency.
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()

        if not ref_tokens or not hyp_tokens:
            return 0.0

        smoothing = SmoothingFunction().method1
        score = sentence_bleu(
            [ref_tokens],
            hyp_tokens,
            weights=(0.5, 0.5),  # unigram + bigram
            smoothing_function=smoothing,
        )
        return round(score, 4)

    except ImportError:
        logger.warning("NLTK not available for BLEU scoring")
        return _simple_overlap(reference, hypothesis)
    except Exception as e:
        logger.error(f"BLEU computation failed: {e}")
        return 0.0


def compute_rouge(reference: str, hypothesis: str) -> Dict[str, float]:
    """
    Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L).
    Falls back to simple overlap if rouge-score is not installed.
    """
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])
        scores = scorer.score(reference, hypothesis)

        return {
            "rouge1": round(scores["rouge1"].fmeasure, 4),
            "rouge2": round(scores["rouge2"].fmeasure, 4),
            "rougeL": round(scores["rougeL"].fmeasure, 4),
        }

    except ImportError:
        logger.warning("rouge-score not available, using simple overlap")
        overlap = _simple_overlap(reference, hypothesis)
        return {"rouge1": overlap, "rouge2": 0.0, "rougeL": overlap}
    except Exception as e:
        logger.error(f"ROUGE computation failed: {e}")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def _simple_overlap(text1: str, text2: str) -> float:
    """Simple word overlap ratio as fallback metric."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    overlap = len(words1 & words2)
    return round(overlap / max(len(words1), len(words2)), 4)


def evaluate_answer(
    query: str,
    answer: str,
    source_texts: List[str],
    reference_answer: Optional[str] = None,
) -> Dict:
    """
    Evaluate the quality of a generated answer.

    Metrics:
    - Source overlap: How much of the answer comes from sources
    - BLEU score (if reference provided)
    - ROUGE score (if reference provided)
    - Length ratio: Answer length vs source length

    Args:
        query: The original query.
        answer: The generated answer.
        source_texts: List of source texts used.
        reference_answer: Optional ground truth for comparison.

    Returns:
        Dict with evaluation metrics.
    """
    metrics = {}

    # Source overlap
    combined_sources = " ".join(source_texts)
    metrics["source_overlap"] = _simple_overlap(answer, combined_sources)

    # Query relevance
    metrics["query_relevance"] = _simple_overlap(query, answer)

    # Length ratio
    if combined_sources:
        metrics["length_ratio"] = round(len(answer) / len(combined_sources), 4)
    else:
        metrics["length_ratio"] = 0.0

    # Comparison metrics (if reference available)
    if reference_answer:
        metrics["bleu"] = compute_bleu(reference_answer, answer)
        metrics["rouge"] = compute_rouge(reference_answer, answer)

    # Groundedness check: what fraction of answer words appear in sources
    answer_words = set(answer.lower().split())
    source_words = set(combined_sources.lower().split())
    if answer_words:
        grounded = len(answer_words & source_words) / len(answer_words)
        metrics["groundedness"] = round(grounded, 4)
    else:
        metrics["groundedness"] = 0.0

    return metrics
