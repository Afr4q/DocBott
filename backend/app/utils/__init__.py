"""
Utility functions used across the application.
Logging, file operations, text cleaning, and common helpers.
"""

import hashlib
import logging
import re
import unicodedata
from pathlib import Path
from typing import List, Optional

from app.config import LOG_LEVEL

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    """Create a named logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    return logger


# ──────────────────────────────────────────────
# File Utilities
# ──────────────────────────────────────────────
def compute_file_hash(file_path: str) -> str:
    """Compute SHA-256 hash of a file for deduplication."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha256.update(block)
    return sha256.hexdigest()


def safe_filename(name: str) -> str:
    """Sanitize a filename to be filesystem-safe."""
    # Normalize unicode characters
    name = unicodedata.normalize("NFKD", name)
    # Remove non-alphanumeric chars except dots, dashes, underscores
    name = re.sub(r"[^\w.\-]", "_", name)
    # Collapse multiple underscores
    name = re.sub(r"_+", "_", name)
    return name.strip("_")


# ──────────────────────────────────────────────
# Text Cleaning
# ──────────────────────────────────────────────
def clean_text(text: str) -> str:
    """
    Clean extracted text by removing OCR artifacts,
    excessive whitespace, and control characters.
    """
    if not text:
        return ""

    # Remove null bytes and control characters (keep newlines and tabs)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Fix common OCR artifacts
    text = text.replace("|", "I")  # common OCR misread
    text = re.sub(r"[''`]", "'", text)  # normalize quotes
    text = re.sub(r'["""]', '"', text)

    # Collapse multiple spaces (but preserve newlines)
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse more than 2 consecutive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in text.split("\n")]
    text = "\n".join(lines)

    return text.strip()


def remove_header_footer(text: str, max_lines: int = 3) -> str:
    """
    Remove likely header/footer lines (repeated short lines at page boundaries).
    Uses heuristic: lines < 60 chars at the start/end of a page.
    """
    lines = text.split("\n")
    if len(lines) <= max_lines * 2:
        return text

    # Remove short lines at start
    start = 0
    for i in range(min(max_lines, len(lines))):
        if len(lines[i].strip()) < 60 and not any(c.isalpha() for c in lines[i][:5]):
            start = i + 1
        else:
            break

    # Remove short lines at end
    end = len(lines)
    for i in range(len(lines) - 1, max(len(lines) - max_lines - 1, 0), -1):
        if len(lines[i].strip()) < 60 and lines[i].strip().isdigit():
            end = i
        else:
            break

    return "\n".join(lines[start:end])


def clean_chunk_for_display(text: str) -> str:
    """
    Deep-clean a text chunk before showing it as part of an answer.
    Removes page numbers, TOC lines, reference lists, headers/footers,
    URL-only lines, and other PDF extraction artifacts.
    """
    if not text:
        return ""

    # Fix hyphenated line-breaks (word- \n continuation → wordcontinuation)
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)

    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        # Skip blank lines
        if not line:
            continue

        # Skip standalone page numbers
        if re.fullmatch(r"[-–—]?\s*\d{1,4}\s*[-–—]?", line):
            continue

        # Skip "Page X of Y" / "Page X" patterns
        if re.match(r"(?i)^page\s+\d+", line):
            continue

        # Skip "1 | P a g e" style
        if re.match(r"\d+\s*\|\s*[Pp]\s*[Aa]\s*[Gg]\s*[Ee]", line):
            continue

        # Skip Roman numerals alone on a line (i, ii, iii, iv, v, vi, vii, viii, ix, x)
        if re.fullmatch(r"(?i)(x{0,3})(ix|iv|v?i{0,3})", line):
            continue

        # Skip Table of Contents lines with dots (e.g. "Chapter 1 ......... 5")
        if re.search(r"\.{4,}", line) and re.search(r"\d", line):
            continue

        # Skip lines that are ALL dots/dashes/underscores
        if re.fullmatch(r"[.\-_\s]{3,}", line):
            continue

        # Skip reference list items (e.g. "[1] Smith, J. (2020)...")
        if re.match(r"^\[\d+\]\s+", line):
            continue

        # Skip footnote-style lines (e.g. "1. Author et al.")
        if re.match(r"^\d{1,2}\.\s+[A-Z][a-z]+", line) and len(line) < 100:
            # Only skip if it looks like a citation, not content
            if any(kw in line.lower() for kw in ["et al", "journal", "volume", "press", "doi", "isbn", "pp.", "vol."]):
                continue

        # Skip URL-only lines
        if re.match(r"^https?://\S+$", line) or re.match(r"^www\.\S+$", line):
            continue

        # Skip copyright/confidential lines
        if re.match(r"(?i)^(copyright|©|all rights reserved|confidential|proprietary|draft)", line):
            continue

        # Skip very short meaningless lines (fewer than 4 chars or only special chars)
        if len(line) < 4:
            continue
        if not any(c.isalpha() for c in line):
            continue

        cleaned_lines.append(line)

    result = " ".join(cleaned_lines)

    # Collapse multiple spaces
    result = re.sub(r"  +", " ", result)

    return result.strip()


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex-based heuristics.
    More robust than simple period splitting for academic text.
    """
    # Handle abbreviations to avoid false splits
    text = re.sub(r"(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|etc|vs|Fig|Eq)\.", r"\1<DOT>", text)
    text = re.sub(r"(\d)\.", r"\1<DOT>", text)  # decimal numbers

    # Split on sentence-ending punctuation followed by space + uppercase
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)

    # Restore dots
    sentences = [s.replace("<DOT>", ".") for s in sentences]

    # Filter empty sentences
    return [s.strip() for s in sentences if s.strip()]


def estimate_confidence(scores: List[float]) -> float:
    """
    Estimate answer confidence from retrieval scores.
    Uses weighted average with diminishing returns.
    """
    if not scores:
        return 0.0

    # Normalize scores to 0-1 range
    max_score = max(scores) if max(scores) > 0 else 1.0
    normalized = [s / max_score for s in scores]

    # Weighted average - top results matter more
    weights = [1.0 / (i + 1) for i in range(len(normalized))]
    total_weight = sum(weights)
    weighted_sum = sum(w * s for w, s in zip(weights, normalized))

    confidence = weighted_sum / total_weight if total_weight > 0 else 0.0
    return round(min(confidence, 1.0), 4)


def truncate_text(text: str, max_length: int = 500) -> str:
    """Truncate text to max_length, ending at a sentence boundary if possible."""
    if len(text) <= max_length:
        return text

    truncated = text[:max_length]
    # Try to end at the last sentence boundary
    last_period = truncated.rfind(".")
    if last_period > max_length * 0.5:
        return truncated[:last_period + 1]
    return truncated + "..."
