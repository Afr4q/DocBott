"""
AI Summarization module - Local HuggingFace-based abstractive summarization.
Uses facebook/bart-large-cnn with direct generate() calls (no pipeline shortcuts).
This is OPTIONAL and togglable - never replaces the grounded extractive answer.
"""

from typing import Optional

from app.config import SUMMARIZATION_MODEL, ENABLE_AI_SUMMARY, OLLAMA_BASE_URL, OLLAMA_MODEL
from app.utils import get_logger, truncate_text, split_into_sentences

logger = get_logger(__name__)

# ──────────────────────────────────────────────
# Model (lazy-loaded)
# ──────────────────────────────────────────────
_summarization_model = None
_summarization_tokenizer = None


def _load_summarization_model():
    """Lazy-load the summarization model and tokenizer."""
    global _summarization_model, _summarization_tokenizer

    if _summarization_model is None:
        logger.info(f"Loading summarization model: {SUMMARIZATION_MODEL}")
        try:
            from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

            _summarization_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL)
            _summarization_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZATION_MODEL)
            logger.info("Summarization model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load summarization model: {e}")
            raise

    return _summarization_model, _summarization_tokenizer


def summarize_text(
    text: str,
    max_length: int = 200,
    min_length: int = 50,
) -> str:
    """
    Generate an abstractive summary using the local BART model.
    Uses direct model.generate() - NOT pipeline shortcuts.

    Args:
        text: The text to summarize.
        max_length: Maximum summary length in tokens.
        min_length: Minimum summary length in tokens.

    Returns:
        Generated summary text.
    """
    if not ENABLE_AI_SUMMARY:
        return ""

    if not text or len(text.strip()) < 50:
        return text.strip()

    try:
        model, tokenizer = _load_summarization_model()

        # Tokenize input - truncate to model's max length
        inputs = tokenizer(
            text,
            max_length=1024,
            truncation=True,
            return_tensors="pt",
        )

        # Generate summary using direct generate() call
        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            min_length=min_length,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

        # Decode output
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary.strip()

    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        return f"[Summarization error: {str(e)}]"


def summarize_for_comparison(
    extractive_answer: str,
    context: str,
    query: str = "",
) -> dict:
    """
    Generate both extractive and AI-generated answers for side-by-side comparison.

    Returns:
        Dict with 'direct_answer' (extractive) and 'ai_answer' (abstractive).
    """
    result = {
        "direct_answer": extractive_answer,
        "ai_answer": "",
        "comparison_available": False,
    }

    if not ENABLE_AI_SUMMARY:
        return result

    try:
        # Prepare context for summarization
        prompt = context
        if query:
            prompt = f"Question: {query}\n\nContext: {context}"

        ai_summary = summarize_text(prompt)

        if ai_summary and ai_summary != extractive_answer:
            result["ai_answer"] = ai_summary
            result["comparison_available"] = True

    except Exception as e:
        logger.error(f"Comparison generation failed: {e}")

    return result


# ──────────────────────────────────────────────
# Direct AI Answer (no PDF grounding)
# ──────────────────────────────────────────────
def direct_ai_answer(query: str, context: str = "") -> str:
    """
    Generate a direct AI answer to a user question.
    First tries Ollama (local LLM), then falls back to BART summarization.
    This is NOT grounded in documents – it's a pure AI response.

    Args:
        query: The user's question.
        context: Optional context from PDF retrieval to supplement the answer.

    Returns:
        AI-generated answer string.
    """
    # Try Ollama first (better for Q&A)
    if context:
        prompt = (
            f"Answer the following question using the provided context.\n\n"
            f"Question: {query}\n\n"
            f"Context: {context}\n\n"
            f"Provide a clear, comprehensive answer:"
        )
    else:
        prompt = (
            f"Answer the following question clearly and comprehensively.\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )

    ollama_response = query_ollama(
        prompt=prompt,
        system_prompt=(
            "You are a knowledgeable AI assistant. Provide clear, accurate, and "
            "well-structured answers. If you are unsure about something, say so. "
            "Use markdown formatting for readability."
        ),
        max_tokens=800,
    )

    if ollama_response and len(ollama_response.strip()) > 20:
        return ollama_response.strip()

    # Fallback: use BART summarization on the context if available
    if context and len(context.strip()) > 50:
        try:
            summary = summarize_text(context, max_length=300, min_length=80)
            if summary:
                return summary
        except Exception as e:
            logger.error(f"BART fallback failed: {e}")

    # Final fallback: acknowledge limitations
    return (
        "I'm unable to generate a direct AI answer at this time. "
        "This feature works best when an AI model like Ollama (LLaMA) is running locally. "
        "You can still get answers grounded in your uploaded PDFs using the PDF answer mode."
    )


# ──────────────────────────────────────────────
# Ollama Integration (Optional, for LLaMA 3)
# ──────────────────────────────────────────────
def query_ollama(
    prompt: str,
    system_prompt: str = "",
    max_tokens: int = 500,
) -> Optional[str]:
    """
    Query a local Ollama instance (e.g., LLaMA 3).
    This is OPTIONAL and used only if Ollama is running.

    Args:
        prompt: The user prompt.
        system_prompt: System-level instruction.
        max_tokens: Maximum response length.

    Returns:
        Generated text or None if Ollama is unavailable.
    """
    try:
        import requests

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "system": system_prompt or (
                    "You are a document analysis assistant. "
                    "Answer ONLY based on the provided context. "
                    "If the answer is not in the context, say so."
                ),
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.3,
                },
                "stream": False,
            },
            timeout=60,
        )

        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            logger.warning(f"Ollama returned status {response.status_code}")
            return None

    except requests.exceptions.ConnectionError:
        logger.debug("Ollama not available (connection refused)")
        return None
    except Exception as e:
        logger.error(f"Ollama query failed: {e}")
        return None


# ──────────────────────────────────────────────
# Cross-Lingual: Language Detection & Translation
# ──────────────────────────────────────────────
import re as _re_ai

def detect_language(text: str) -> str:
    """
    Detect the language of the given text.
    Returns ISO 639-1 code (e.g., 'en', 'es', 'fr', 'hi').
    Uses a scoring approach to avoid false positives from short substring matches.
    """
    if not text or len(text.strip()) < 10:
        return "en"

    try:
        # Use character-range heuristics first for non-Latin scripts
        sample = text[:1000]

        # Devanagari (Hindi, Sanskrit, Marathi)
        if _re_ai.search(r'[\u0900-\u097F]', sample):
            return "hi"
        # Arabic
        if _re_ai.search(r'[\u0600-\u06FF]', sample):
            return "ar"
        # CJK (Chinese)
        if _re_ai.search(r'[\u4e00-\u9fff]', sample):
            return "zh"
        # Japanese (Hiragana/Katakana)
        if _re_ai.search(r'[\u3040-\u30ff]', sample):
            return "ja"
        # Korean
        if _re_ai.search(r'[\uac00-\ud7af]', sample):
            return "ko"
        # Cyrillic (Russian)
        if _re_ai.search(r'[\u0400-\u04FF]', sample):
            return "ru"
        # Thai
        if _re_ai.search(r'[\u0e00-\u0e7f]', sample):
            return "th"

        # For Latin-script languages, use word-boundary matching with scoring
        # to avoid false positives from English text containing "la", "le", etc.
        lower = sample.lower()
        words = _re_ai.findall(r'\b[a-zA-ZÀ-ÿ]+\b', lower)
        total_words = len(words) if words else 1
        word_set = set(words)

        # Strong French indicators (require word boundaries, not substrings)
        fr_markers = {'le', 'la', 'les', 'une', 'des', 'est', 'sont', 'avec',
                      'pour', 'dans', 'sur', 'que', 'qui', 'pas', 'nous',
                      'vous', 'leur', 'mais', 'comme', 'tout', 'bien',
                      'aussi', 'cette', 'ces', 'ont', 'fait', 'peut', 'entre'}
        # English words that overlap with French (should NOT count for French)
        fr_false_positives = {'la', 'le', 'des', 'pour', 'sur', 'entre', 'est'}

        es_markers = {'el', 'los', 'las', 'una', 'del', 'por', 'con', 'como',
                      'pero', 'sin', 'sobre', 'entre', 'hasta', 'desde',
                      'puede', 'tiene', 'hace', 'todo', 'esta', 'cada', 'muy',
                      'donde', 'cuando', 'porque', 'también', 'otro', 'fueron'}
        es_false_positives = {'el', 'con', 'como', 'entre', 'sin'}

        de_markers = {'der', 'die', 'das', 'und', 'ist', 'ein', 'eine', 'den',
                      'dem', 'des', 'auf', 'mit', 'sich', 'nicht', 'auch',
                      'noch', 'nach', 'wird', 'bei', 'oder', 'kann', 'aber',
                      'nur', 'wenn', 'als', 'aus', 'wie', 'haben', 'werden'}

        pt_markers = {'uma', 'são', 'não', 'com', 'para', 'mais', 'pelo',
                      'pela', 'nos', 'nas', 'dos', 'das', 'quando', 'mesmo',
                      'ainda', 'depois', 'onde', 'pode', 'muito', 'todos',
                      'essa', 'esse', 'isso', 'aqui', 'então', 'também'}
        pt_false_positives = {'para', 'com', 'nos'}

        # English common words (to boost English detection)
        en_markers = {'the', 'and', 'is', 'are', 'was', 'were', 'have', 'has',
                      'been', 'will', 'would', 'could', 'should', 'can', 'do',
                      'does', 'did', 'not', 'but', 'from', 'they', 'their',
                      'which', 'what', 'when', 'where', 'how', 'who', 'this',
                      'that', 'these', 'those', 'there', 'here', 'each',
                      'every', 'some', 'any', 'other', 'such', 'than', 'then',
                      'only', 'very', 'also', 'into', 'through', 'between',
                      'after', 'before', 'because', 'while', 'being', 'about'}

        lang_scores = {}

        # Score each language
        en_hits = len(word_set & en_markers)
        lang_scores['en'] = en_hits / total_words

        # For non-English, only count strong markers (exclude false positives if English score is high)
        fr_strong = word_set & (fr_markers - fr_false_positives) if en_hits > 3 else word_set & fr_markers
        lang_scores['fr'] = len(fr_strong) / total_words

        es_strong = word_set & (es_markers - es_false_positives) if en_hits > 3 else word_set & es_markers
        lang_scores['es'] = len(es_strong) / total_words

        lang_scores['de'] = len(word_set & de_markers) / total_words

        pt_strong = word_set & (pt_markers - pt_false_positives) if en_hits > 3 else word_set & pt_markers
        lang_scores['pt'] = len(pt_strong) / total_words

        # Require non-English languages to have a much higher score than English
        # to prevent false positives on technical English text
        best_lang = max(lang_scores, key=lang_scores.get)
        best_score = lang_scores[best_lang]

        if best_lang != 'en' and best_score > lang_scores['en'] * 1.5 and best_score > 0.02:
            return best_lang

        # Also check for accented characters that are strong indicators
        accent_count = len(_re_ai.findall(r'[àâäéèêëïîôùûüçœæ]', lower))
        if accent_count > 5 and lang_scores.get('fr', 0) > lang_scores.get('en', 0):
            return "fr"

        umlaut_count = len(_re_ai.findall(r'[äöüß]', lower))
        if umlaut_count > 3:
            return "de"

        tilde_count = len(_re_ai.findall(r'[ñ¿¡]', lower))
        if tilde_count > 2:
            return "es"

        cedilla_count = len(_re_ai.findall(r'[ãõç]', lower))
        if cedilla_count > 3 and lang_scores.get('pt', 0) > lang_scores.get('fr', 0):
            return "pt"

        return "en"
    except Exception:
        return "en"


def translate_text(text: str, target_lang: str = "en", source_lang: str = "") -> str:
    """
    Translate text to target language using Ollama.
    Falls back to returning original text if translation unavailable.
    """
    if not text or target_lang == source_lang:
        return text

    lang_names = {
        "en": "English", "es": "Spanish", "fr": "French", "de": "German",
        "hi": "Hindi", "ar": "Arabic", "zh": "Chinese", "ja": "Japanese",
        "ko": "Korean", "ru": "Russian", "pt": "Portuguese", "it": "Italian",
        "th": "Thai", "nl": "Dutch", "sv": "Swedish",
    }
    target_name = lang_names.get(target_lang, target_lang)
    source_name = lang_names.get(source_lang, "the source language") if source_lang else "its original language"

    prompt = (
        f"Translate the following text from {source_name} to {target_name}. "
        f"Return ONLY the translation, nothing else.\n\n"
        f"Text: {text[:2000]}\n\nTranslation:"
    )

    result = query_ollama(prompt, system_prompt="You are a professional translator. Translate accurately.", max_tokens=1000)
    if result and len(result.strip()) > 5:
        return result.strip()
    return text  # fallback: return original


# ──────────────────────────────────────────────
# Fact-Checking Against External Knowledge
# ──────────────────────────────────────────────
def fact_check_answer(answer: str, query: str, sources: list = None) -> dict:
    """
    Fact-check an answer by analyzing consistency and suggesting verification points.
    Uses AI to identify claims and assess confidence.
    """
    source_info = ""
    if sources:
        source_info = "\n".join(f"- {s.get('file', 'Document')}, Page {s.get('page', '?')}: {s.get('content_preview', '')[:200]}" for s in sources[:3])

    prompt = (
        f"Analyze the following answer for factual accuracy. Identify key claims, "
        f"rate their verifiability, and suggest what to verify.\n\n"
        f"Question: {query}\n\n"
        f"Answer: {answer}\n\n"
        f"Source extracts:\n{source_info}\n\n"
        f"Respond in JSON format with these fields:\n"
        f'- "claims": list of key factual claims found\n'
        f'- "confidence_rating": "high", "medium", or "low"\n'
        f'- "verification_notes": list of things to verify\n'
        f'- "consistency": "consistent", "partially_consistent", or "inconsistent" (how well answer matches sources)\n'
        f'- "summary": short overall assessment\n'
    )

    result = query_ollama(
        prompt,
        system_prompt="You are a fact-checking assistant. Analyze claims carefully. Always respond in valid JSON.",
        max_tokens=600,
    )

    if result:
        try:
            import json
            # Try to parse JSON from the response
            # Handle cases where response has extra text around JSON
            json_match = _re_ai.search(r'\{.*\}', result, _re_ai.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return {
                    "claims": parsed.get("claims", []),
                    "confidence_rating": parsed.get("confidence_rating", "medium"),
                    "verification_notes": parsed.get("verification_notes", []),
                    "consistency": parsed.get("consistency", "partially_consistent"),
                    "summary": parsed.get("summary", "Analysis completed."),
                    "available": True,
                }
        except (json.JSONDecodeError, AttributeError):
            pass

    # Fallback: basic heuristic analysis
    claims = [s.strip() for s in split_into_sentences(answer) if len(s.strip()) > 20][:5]
    return {
        "claims": claims,
        "confidence_rating": "medium",
        "verification_notes": ["AI fact-checker unavailable. Manual verification recommended."],
        "consistency": "unknown",
        "summary": "Automated fact-check limited. The answer is based on uploaded document content.",
        "available": False,
    }


# ──────────────────────────────────────────────
# FAQ Generator
# ──────────────────────────────────────────────
def generate_faqs_from_text(text: str, num_questions: int = 5) -> list:
    """
    Generate FAQ-style Q&A pairs from document text.
    Returns list of dicts with 'question' and 'answer' keys.
    """
    # Clean the text — remove excessive whitespace, short lines, page markers
    lines = [l.strip() for l in text.split('\n') if len(l.strip()) > 15]
    cleaned = '\n'.join(lines)[:4000]

    if len(cleaned) < 50:
        return []

    prompt = (
        f"You are reading a document. Based ONLY on the content below, generate exactly {num_questions} "
        f"frequently asked questions (FAQs) that someone studying this document would ask.\n\n"
        f"RULES:\n"
        f"- Each question MUST be answerable using ONLY the text below\n"
        f"- Each answer MUST quote or closely paraphrase specific facts from the text\n"
        f"- Do NOT invent information not present in the text\n"
        f"- Do NOT ask meta questions like 'What is this document about?'\n"
        f"- Ask specific, factual questions about key concepts, data, definitions, processes, or conclusions in the text\n"
        f"- Keep answers concise (1-3 sentences)\n\n"
        f"DOCUMENT TEXT:\n---\n{cleaned}\n---\n\n"
        f"Respond ONLY with a JSON array of {num_questions} objects, each with \"question\" and \"answer\" fields."
    )

    result = query_ollama(
        prompt,
        system_prompt="You are a precise FAQ generator. You create Q&A pairs strictly from the provided document text. Never fabricate information. Respond ONLY in valid JSON array format.",
        max_tokens=1500,
    )

    if result:
        try:
            import json
            json_match = _re_ai.search(r'\[.*\]', result, _re_ai.DOTALL)
            if json_match:
                faqs = json.loads(json_match.group())
                valid = []
                for faq in faqs:
                    q = faq.get("question", "").strip()
                    a = faq.get("answer", "").strip()
                    if q and a and len(q) > 10 and len(a) > 10:
                        valid.append({"question": q, "answer": a})
                if valid:
                    return valid[:num_questions]
        except (json.JSONDecodeError, AttributeError):
            pass

    # Fallback: extract key sentences and form Q&A from them
    try:
        sentences = [s.strip() for s in split_into_sentences(cleaned) if len(s.strip()) > 40]
        if not sentences:
            return []

        faqs = []
        seen = set()
        for sent in sentences:
            if len(faqs) >= num_questions:
                break
            # Skip duplicate-ish sentences
            key = sent[:50].lower()
            if key in seen:
                continue
            seen.add(key)

            # Extract a meaningful subject from the sentence to form a question
            words = sent.split()
            if len(words) < 6:
                continue

            # Find key noun phrases to ask about
            # Look for patterns: "X is/are Y", "X refers to Y", "X means Y"
            lower = sent.lower()
            if ' is ' in lower or ' are ' in lower or ' was ' in lower or ' were ' in lower:
                split_word = ' is ' if ' is ' in lower else ' are ' if ' are ' in lower else ' was ' if ' was ' in lower else ' were '
                parts = sent.split(split_word, 1)
                subject = parts[0].strip().rstrip(',').strip()
                if 10 < len(subject) < 80:
                    q = f"What is {subject}?"
                    faqs.append({"question": q, "answer": sent})
                    continue

            # Generic: "According to the document, [sentence]?"
            if len(sent) > 40:
                # Extract first meaningful phrase
                subject = ' '.join(words[:6])
                q = f"What does the document state about {subject.rstrip('.,;:')}?"
                faqs.append({"question": q, "answer": sent})

        return faqs
    except Exception:
        return []


# ──────────────────────────────────────────────
# Document Relationship Analysis
# ──────────────────────────────────────────────
def analyze_document_relationships(doc_summaries: dict) -> dict:
    """
    Analyze relationships between multiple documents.
    doc_summaries: {doc_id: {"name": str, "text_sample": str}}
    Returns relationship data for visualization.
    """
    if not doc_summaries or len(doc_summaries) < 2:
        return {"nodes": [], "edges": [], "clusters": []}

    nodes = []
    edges = []

    doc_items = list(doc_summaries.items())

    # Expanded stopwords for better filtering
    stopwords = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'need', 'must',
        'of', 'in', 'to', 'for', 'with', 'on', 'at', 'by', 'from', 'into',
        'through', 'during', 'before', 'after', 'between', 'under', 'above',
        'and', 'or', 'but', 'not', 'nor', 'so', 'yet', 'both', 'either',
        'this', 'that', 'these', 'those', 'it', 'its', 'their', 'there',
        'which', 'what', 'who', 'whom', 'whose', 'where', 'when', 'how',
        'each', 'every', 'all', 'any', 'some', 'no', 'more', 'most', 'other',
        'than', 'then', 'also', 'such', 'like', 'just', 'only', 'very',
        'about', 'over', 'out', 'up', 'down', 'off', 'if', 'because', 'as',
        'while', 'however', 'although', 'since', 'until', 'unless',
        'been', 'being', 'here', 'come', 'came', 'make', 'made', 'well',
        'back', 'even', 'still', 'way', 'take', 'taken', 'many', 'much',
        'used', 'using', 'use', 'based', 'one', 'two', 'new', 'first',
        'may', 'part', 'see', 'now', 'get', 'also', 'set', 'per', 'end',
        'page', 'figure', 'table', 'chapter', 'section', 'note', 'ref',
    }

    def extract_keywords(text_sample):
        """Extract meaningful keywords from text, stripping punctuation and noise."""
        if not text_sample:
            return set()
        # Remove punctuation, normalize
        cleaned = _re_ai.sub(r'[^a-zA-Z\s]', ' ', text_sample.lower())
        words = cleaned.split()
        # Filter: not stopword, length > 3, not all same char, not a number-like word
        keywords = set()
        for w in words:
            w = w.strip()
            if (len(w) > 3 and
                w not in stopwords and
                not w.isdigit() and
                len(set(w)) > 1):  # skip 'aaaa'-like tokens
                keywords.add(w)
        return keywords

    # Build keyword sets for each document
    doc_keywords = {}
    for doc_id, info in doc_items:
        text = info.get("text_sample", "")
        kws = extract_keywords(text)
        doc_keywords[doc_id] = kws
        nodes.append({
            "id": doc_id,
            "label": info.get("name", f"Document {doc_id}"),
            "keyword_count": len(kws),
        })

    # Compare each pair using Jaccard similarity on keywords
    for i in range(len(doc_items)):
        id_a = doc_items[i][0]
        words_a = doc_keywords.get(id_a, set())
        if len(words_a) < 3:  # Skip documents with too few keywords
            continue

        for j in range(i + 1, len(doc_items)):
            id_b = doc_items[j][0]
            words_b = doc_keywords.get(id_b, set())
            if len(words_b) < 3:
                continue

            overlap = words_a & words_b
            union = words_a | words_b

            if not union:
                continue

            jaccard = len(overlap) / len(union)

            # Also compute overlap coefficient (for cases where one doc is much smaller)
            overlap_coeff = len(overlap) / min(len(words_a), len(words_b)) if min(len(words_a), len(words_b)) > 0 else 0

            # Use the higher of the two metrics
            similarity = max(jaccard, overlap_coeff * 0.7)

            if similarity > 0.03 and len(overlap) >= 2:  # At least 3% similar AND 2+ shared words
                # Rank shared topics — prefer longer (more specific) words
                shared_sorted = sorted(overlap, key=lambda w: -len(w))
                shared_topics = shared_sorted[:8]

                if similarity > 0.15:
                    strength = "strong"
                elif similarity > 0.07:
                    strength = "moderate"
                else:
                    strength = "weak"

                edges.append({
                    "source": id_a,
                    "target": id_b,
                    "similarity": round(similarity, 3),
                    "shared_topics": shared_topics,
                    "shared_count": len(overlap),
                    "relationship": strength,
                })

    # Sort edges by similarity descending
    edges.sort(key=lambda e: e["similarity"], reverse=True)

    return {"nodes": nodes, "edges": edges}


# ──────────────────────────────────────────────
# Confidence Explainer
# ──────────────────────────────────────────────
def explain_confidence(confidence: float, sources: list, query: str) -> dict:
    """
    Generate a human-readable explanation of why a confidence score was given.
    """
    factors = []

    # Source count factor
    source_count = len(sources) if sources else 0
    if source_count >= 3:
        factors.append({"factor": "Multiple sources found", "impact": "positive", "detail": f"{source_count} relevant sources identified"})
    elif source_count == 0:
        factors.append({"factor": "No sources found", "impact": "negative", "detail": "No matching content in documents"})
    else:
        factors.append({"factor": "Limited sources", "impact": "neutral", "detail": f"Only {source_count} source(s) found"})

    # Score quality
    if sources:
        top_score = max(s.get("score", 0) for s in sources)
        if top_score > 0.8:
            factors.append({"factor": "High relevance match", "impact": "positive", "detail": f"Best match: {(top_score*100):.0f}%"})
        elif top_score > 0.5:
            factors.append({"factor": "Moderate relevance", "impact": "neutral", "detail": f"Best match: {(top_score*100):.0f}%"})
        else:
            factors.append({"factor": "Low relevance scores", "impact": "negative", "detail": f"Best match: {(top_score*100):.0f}%"})

        # Source diversity (multiple documents)
        unique_files = set(s.get("file", "") for s in sources)
        if len(unique_files) > 1:
            factors.append({"factor": "Cross-document validation", "impact": "positive", "detail": f"Found in {len(unique_files)} different documents"})

    # Confidence level
    level = "high" if confidence > 0.7 else "medium" if confidence > 0.4 else "low"

    return {
        "score": confidence,
        "level": level,
        "factors": factors,
        "recommendation": (
            "This answer is well-supported by your documents." if level == "high" else
            "Consider verifying this answer with additional sources." if level == "medium" else
            "This answer has limited support. Upload more relevant documents or rephrase your question."
        ),
    }
