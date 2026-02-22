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
def _extract_relevant_sentences(query: str, context: str, top_n: int = 8) -> str:
    """
    Extract the most query-relevant sentences from context using keyword overlap scoring.
    Used as a smart pre-filter before passing text to BART.
    """
    import re
    # Tokenise into sentences
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', context) if len(s.strip()) > 20]
    if not sentences:
        return context[:1024]

    # Build a simple query keyword set (skip stopwords)
    stopwords = {
        'the','a','an','is','are','was','were','be','been','being','have','has',
        'had','do','does','did','will','would','could','should','may','might',
        'of','in','to','and','or','for','on','at','by','with','from','this',
        'that','these','those','what','which','how','when','where','who','why'
    }
    query_words = {w.lower() for w in re.findall(r'\b\w+\b', query) if w.lower() not in stopwords and len(w) > 2}

    # Score each sentence
    def _score(sent: str) -> float:
        words = {w.lower() for w in re.findall(r'\b\w+\b', sent)}
        overlap = len(query_words & words)
        # Boost sentences that contain key query terms early
        return overlap + (0.5 if any(q in sent.lower() for q in query_words) else 0)

    scored = sorted(enumerate(sentences), key=lambda x: _score(x[1]), reverse=True)
    # Take top_n in original order for coherence
    top_indices = sorted(i for i, _ in scored[:top_n])
    return " ".join(sentences[i] for i in top_indices)


def direct_ai_answer(query: str, context: str = "") -> str:
    """
    Generate a direct AI answer to a user question.
    1. Tries Ollama (LLaMA) first.
    2. Falls back to BART with query-focused context selection.
    3. Falls back to extractive answer from top-scored sentences.
    """
    # ── 1. Try Ollama ──────────────────────────────────────────
    if context:
        prompt = (
            f"You are an expert assistant. Answer the question using ONLY the provided context.\n\n"
            f"Question: {query}\n\n"
            f"Context:\n{context[:2000]}\n\n"
            f"Provide a thorough, well-structured answer in 3-5 sentences:"
        )
    else:
        prompt = (
            f"Answer the following question clearly, accurately and in detail.\n\n"
            f"Question: {query}\n\nAnswer:"
        )

    ollama_response = query_ollama(
        prompt=prompt,
        system_prompt=(
            "You are a knowledgeable AI assistant. Provide clear, accurate, "
            "well-structured answers. Use markdown formatting when helpful."
        ),
        max_tokens=800,
    )
    if ollama_response and len(ollama_response.strip()) > 30:
        return ollama_response.strip()

    # ── 2. BART fallback: focus context on the query first ─────
    if context and len(context.strip()) > 50:
        try:
            # Select the most query-relevant sentences before feeding BART
            focused = _extract_relevant_sentences(query, context, top_n=8)
            bart_prompt = f"Question: {query}\n\nContext: {focused}"
            summary = summarize_text(bart_prompt, max_length=250, min_length=60)
            if summary and len(summary.strip()) > 30:
                return summary.strip()
        except Exception as e:
            logger.error(f"BART Q&A fallback failed: {e}")

    # ── 3. Extractive fallback: return the top-scored sentences verbatim ──
    if context and len(context.strip()) > 50:
        try:
            relevant = _extract_relevant_sentences(query, context, top_n=4)
            if relevant and len(relevant.strip()) > 30:
                return (
                    f"Based on the document content:\n\n{relevant.strip()}\n\n"
                    f"*(Note: For richer AI-generated explanations, start Ollama locally — "
                    f"`ollama run llama3`)*"
                )
        except Exception:
            pass

    # ── 4. Final graceful fallback ─────────────────────────────
    return (
        "The AI answer mode works best when an Ollama model is running locally. "
        "Switch to **PDF mode** to get answers grounded in your uploaded documents, "
        "or start Ollama with: `ollama run llama3`"
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
    Uses Ollama (if available) or BART-powered topic extraction fallback.
    Returns list of dicts with 'question' and 'answer' keys.
    """
    # ── Internal PDF/slide noise cleaning ─────────────────────────────────────
    text = _re_ai.sub(r'\[/?TABLE\]', ' ', text, flags=_re_ai.IGNORECASE)
    text = _re_ai.sub(r'[^\n]*(?:\|[^\n]*){2,}', ' ', text)
    text = _re_ai.sub(r'\s*\|\s*', ' ', text)
    text = _re_ai.sub(r'\b\d{1,3}\s*/\s*\d{2,3}\b', ' ', text)
    text = _re_ai.sub(r'[-=_]{3,}', ' ', text)
    text = _re_ai.sub(r'\b[A-Z]{2,4}\s*\d{3,5}\b', ' ', text)
    text = _re_ai.sub(r'\(\d-\d-\d\)', ' ', text)
    text = _re_ai.sub(r' {2,}', ' ', text).strip()

    # Filter sentences — skip metadata/headers
    _META_SKIP = [
        r'^(?:lecture\s+notes?|prepared\s+by|department\s+of|college\s+of)',
        r'^(?:module|unit|chapter|syllabus|references?|bibliography)',
        r'^(?:prof\b|dr\.?\s|mr\.?\s)',
        r'(?:all\s+rights?\s+reserved|copyright|ISBN)',
    ]
    raw_sents = [s.strip() for s in _re_ai.split(r'(?<=[.!?])\s+', text)]
    sentences = []
    seen_keys: set = set()
    for s in raw_sents:
        words = s.split()
        if len(words) < 8:
            continue
        alpha = _re_ai.sub(r'[^a-zA-Z]', '', s)
        if alpha and sum(1 for c in alpha if c.isupper()) / len(alpha) > 0.7:
            continue
        if any(_re_ai.search(p, s, _re_ai.IGNORECASE) for p in _META_SKIP):
            continue
        key = s[:50].lower()
        if key in seen_keys:
            continue
        seen_keys.add(key)
        sentences.append(s)

    cleaned = ' '.join(sentences)[:6000]
    if len(cleaned) < 80:
        return []

    # ── Try Ollama first ──────────────────────────────────────────────────────
    prompt = (
        f"Based ONLY on the text below, generate exactly {num_questions} "
        f"study-oriented FAQ questions and answers.\n\n"
        f"RULES:\n"
        f"- Ask about definitions, processes, comparisons, advantages, applications\n"
        f"- Each answer should be detailed (2-4 sentences) using facts from the text\n"
        f"- Do NOT ask about authors, page numbers, or document metadata\n"
        f"- Questions should be specific and educational\n\n"
        f"TEXT:\n{cleaned}\n\n"
        f"Respond ONLY with a JSON array of objects with \"question\" and \"answer\" fields."
    )
    result = query_ollama(
        prompt,
        system_prompt="You are a study FAQ generator. Create specific, educational Q&A pairs from document text. Answers must be detailed and informative. JSON array output only.",
        max_tokens=2000,
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

    # ── Fallback: Smart topic extraction + BART answer generation ─────────────
    _STOP_WORDS = {
        'the','a','an','is','are','was','were','be','been','being','have','has',
        'had','it','its','this','that','these','those','in','on','at','of','for',
        'to','and','or','by','with','from','but','not','can','will','which','so',
        'as','also','such','may','would','could','should','each','than','into',
        'about','between','after','before','through','during','our','we','their',
        'they','then','there','here','when','where','if','how','what','more','most',
        'some','any','all','other','only','very','just','even','still','much',
    }

    # Pronouns / demonstratives that should never start a subject
    _PRONOUN_STRIP = _re_ai.compile(
        r'^(?:such|this|that|these|those|it|its|they|their|our|we|'
        r'some|any|each|every|most|many|few|several|various|other)\s+',
        _re_ai.IGNORECASE,
    )

    def _clean_subject(subj: str) -> str:
        """Strip articles, pronouns, and leading junk from extracted subjects."""
        subj = _re_ai.sub(r'^(the|a|an)\s+', '', subj.strip(), flags=_re_ai.IGNORECASE)
        subj = _PRONOUN_STRIP.sub('', subj)
        subj = subj.strip().rstrip('.,;:')
        return subj

    def _keyterms(txt: str, n: int = 4) -> str:
        ws = [w for w in txt.split() if w.lower() not in _STOP_WORDS and len(w) > 2]
        if not ws:
            return "this concept"
        ws[0] = ws[0].capitalize()
        return " ".join(ws[:n])

    def _get_rich_context(question: str, window_idx: int, window: int = 2) -> str:
        """Get context by combining nearby text + keyword-relevant sentences."""
        # Nearby sentences (local context)
        start = max(0, window_idx - 1)
        end = min(len(sentences), window_idx + window + 1)
        nearby = " ".join(sentences[start:end])

        # Also pull keyword-relevant sentences from the full text
        relevant = _extract_relevant_sentences(question, cleaned, top_n=5)

        # Combine, deduplicate, cap at ~1000 chars
        combined = nearby + " " + relevant
        # Simple dedup: split into sentences and remove duplicates
        all_sents = _re_ai.split(r'(?<=[.!?])\s+', combined)
        unique = []
        seen = set()
        for s in all_sents:
            k = s[:40].lower()
            if k not in seen and len(s.strip()) > 20:
                seen.add(k)
                unique.append(s.strip())
        return " ".join(unique)[:1200]

    try:
        topics: list[dict] = []
        used: set = set()

        def _add(q: str, ctx_idx: int, window: int = 2):
            key = q.lower()[:60]
            if key in used or len(q) < 12:
                return
            # Validate question grammar — skip if it looks broken
            if q.count(' is ') > 1:  # "What is X is done by?"
                return
            used.add(key)
            context = _get_rich_context(q, ctx_idx, window)
            topics.append({"question": q, "context": context})

        # Pass 1: Definitions — "X is a/an/the ...", "X refers to ..."
        for i, s in enumerate(sentences):
            for pat in [
                r'^(.{3,60}?)\s+(?:is|are)\s+(?:a|an|the|one of)\s+(.{15,})',
                r'^(.{3,60}?)\s+(?:refers? to|is defined as|means|is known as|is called)\s+(.{15,})',
            ]:
                m = _re_ai.match(pat, s, _re_ai.IGNORECASE)
                if m:
                    term = _clean_subject(m.group(1))
                    if 1 <= len(term.split()) <= 5 and len(term) > 2:
                        _add(f"What is {term}?", i)
                    break

        # Pass 2: Properties / Features
        for i, s in enumerate(sentences):
            for pat in [
                r'^(.{3,50}?)\s+(?:has|have|contains?|consists? of|includes?)\s+(.{15,})',
                r'^(.{3,50}?)\s+(?:supports?|provides?|allows?|enables?)\s+(.{15,})',
            ]:
                m = _re_ai.match(pat, s, _re_ai.IGNORECASE)
                if m:
                    subj = _clean_subject(m.group(1))
                    if len(subj.split()) <= 5 and len(subj) > 2:
                        _add(f"What are the key features of {subj}?", i)
                    break

        # Pass 3: Complexity / Performance
        for i, s in enumerate(sentences):
            if _re_ai.search(r'O\([^\)]+\)', s) or _re_ai.search(r'\b(worst|average|best)\s+case\b', s, _re_ai.IGNORECASE):
                m = _re_ai.match(r'^(?:The\s+)?(.{3,40}?)(?:\s+(?:is|has|takes?)\b)', s, _re_ai.IGNORECASE)
                subj = _clean_subject(m.group(1)) if m else _keyterms(s, 3)
                _add(f"What is the time/space complexity of {subj}?", i)

        # Pass 4: Comparisons
        for i, s in enumerate(sentences):
            lower = s.lower()
            if any(kw in lower for kw in [' compared to ', ' unlike ', ' versus ', ' difference between ', ' similar to ']):
                _add(f"How does {_keyterms(s, 3)} compare?", i)

        # Pass 5: Advantages / Disadvantages
        for i, s in enumerate(sentences):
            lower = s.lower()
            m = _re_ai.match(r'^(?:The\s+)?(.{3,40}?)(?:\s+(?:is|are|has|have|offers?)\b)', s, _re_ai.IGNORECASE)
            subj = _clean_subject(m.group(1)) if m else _keyterms(s, 3)
            if _re_ai.search(r'\b(advantage|benefit|strength|efficient|faster)\b', lower):
                _add(f"What are the advantages of {subj}?", i)
            elif _re_ai.search(r'\b(disadvantage|drawback|limitation|slow|overhead)\b', lower):
                _add(f"What are the limitations of {subj}?", i)

        # Pass 6: Applications / Usage
        for i, s in enumerate(sentences):
            lower = s.lower()
            if _re_ai.search(r'\b(used (for|in|to)|application|commonly|typically)\b', lower):
                m = _re_ai.match(r'^(?:The\s+)?(.{3,40}?)(?:\s+(?:is|are|can)\b)', s, _re_ai.IGNORECASE)
                subj = _clean_subject(m.group(1)) if m else _keyterms(s, 3)
                _add(f"What are the applications of {subj}?", i)

        # Pass 7: Processes / Algorithms / Steps
        for i, s in enumerate(sentences):
            if _re_ai.search(r'\b(step|first|then|procedure|algorithm|process)\b', s, _re_ai.IGNORECASE):
                m = _re_ai.match(r'^(?:The\s+)?(.{3,40}?)(?:\s+(?:is|works|involves|begins)\b)', s, _re_ai.IGNORECASE)
                subj = _clean_subject(m.group(1)) if m else _keyterms(s, 4)
                _add(f"How does {subj} work?", i, window=3)

        # Pass 8: Types / Categories
        for i, s in enumerate(sentences):
            if _re_ai.search(r'\b(types?\s+of|kinds?\s+of|categor|classified|classification)\b', s, _re_ai.IGNORECASE):
                m = _re_ai.search(r'types?\s+of\s+(.{3,40}?)(?:\s*[:.,]|$)', s, _re_ai.IGNORECASE)
                if m:
                    _add(f"What are the types of {_clean_subject(m.group(1))}?", i)
                else:
                    _add(f"What are the types of {_keyterms(s, 3)}?", i)

        # Pass 9: Examples / Illustrations
        for i, s in enumerate(sentences):
            lower = s.lower()
            if _re_ai.search(r'\b(for example|for instance|e\.g\.|such as|like)\b', lower):
                m = _re_ai.match(r'^(?:The\s+)?(.{3,40}?)(?:\s+(?:is|are|has|include|can)\b)', s, _re_ai.IGNORECASE)
                subj = _clean_subject(m.group(1)) if m else _keyterms(s, 3)
                _add(f"What are examples of {subj}?", i)

        # Pass 10: Purpose / Importance
        for i, s in enumerate(sentences):
            lower = s.lower()
            if any(kw in lower for kw in ['because ', 'in order to ', 'so that ', 'important ', 'essential ', 'crucial ', 'necessary ']):
                m = _re_ai.match(r'^(?:The\s+)?(.{3,40}?)(?:\s+(?:is|are|was|exists?|works?)\b)', s, _re_ai.IGNORECASE)
                subj = _clean_subject(m.group(1)) if m else _keyterms(s, 4)
                _add(f"Why is {subj} important?", i)

        # Pass 11: Fill remaining with concept explanations
        for i, s in enumerate(sentences):
            if len(topics) >= num_questions * 2:
                break
            m = _re_ai.match(r'^(?:The\s+)?(.{3,40}?)(?:\s+(?:is|are|was|were|has|have|can|provides?)\b)', s, _re_ai.IGNORECASE)
            if m:
                subj = _clean_subject(m.group(1))
                if len(subj) > 3 and len(subj.split()) <= 5:
                    _add(f"Explain {subj} in detail.", i, window=3)

        if not topics:
            return []

        # ── Generate detailed answers using BART for each topic ───────────────
        faqs = []
        for topic in topics[:num_questions * 2]:
            if len(faqs) >= num_questions:
                break
            question = topic["question"]
            context = topic["context"]

            answer = None

            # BART pass 1: focused Q&A summarization
            try:
                bart_prompt = f"Question: {question}\n\nContext: {context}"
                polished = summarize_text(bart_prompt, max_length=150, min_length=40)
                if (polished
                    and len(polished.split()) >= 8
                    and "Question:" not in polished
                    and "Context:" not in polished
                    and len(polished) > 40):
                    answer = polished
            except Exception:
                pass

            # BART pass 2: try plain summarization of context if Q&A format failed
            if not answer:
                try:
                    plain = summarize_text(context, max_length=120, min_length=30)
                    if plain and len(plain.split()) >= 8 and len(plain) > 40:
                        answer = plain
                except Exception:
                    pass

            # Fallback: extract the best 2-3 sentences from context
            if not answer:
                ctx_sents = [s.strip() for s in _re_ai.split(r'(?<=[.!?])\s+', context)
                             if len(s.strip()) > 25]
                if len(ctx_sents) >= 2:
                    answer = " ".join(ctx_sents[:3])
                elif ctx_sents:
                    answer = ctx_sents[0]
                else:
                    answer = context[:300]

            if answer and len(answer) > 20:
                faqs.append({"question": question, "answer": answer})

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
