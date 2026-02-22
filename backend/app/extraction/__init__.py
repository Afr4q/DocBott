"""
Extraction module - Digital PDF text extraction using PyMuPDF.
Handles text, metadata, and page-level content extraction.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field

import fitz  # PyMuPDF

from app.utils import clean_text, remove_header_footer, get_logger

logger = get_logger(__name__)


@dataclass
class PageContent:
    """Represents extracted content from a single page."""
    page_number: int
    text: str
    has_images: bool = False
    has_tables: bool = False
    char_count: int = 0
    source_type: str = "text"  # "text", "ocr", "table"


@dataclass
class ExtractionResult:
    """Complete extraction result from a PDF document."""
    pages: List[PageContent] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    page_count: int = 0
    total_chars: int = 0
    has_digital_text: bool = False
    needs_ocr: bool = False
    error: Optional[str] = None


def extract_metadata(doc: fitz.Document) -> Dict:
    """Extract PDF metadata (title, author, subject, etc.)."""
    meta = doc.metadata or {}
    return {
        "title": meta.get("title", ""),
        "author": meta.get("author", ""),
        "subject": meta.get("subject", ""),
        "creator": meta.get("creator", ""),
        "producer": meta.get("producer", ""),
        "page_count": doc.page_count,
        "is_encrypted": doc.is_encrypted,
    }


def extract_page_text(page: fitz.Page) -> str:
    """
    Extract text from a single PDF page.
    Uses PyMuPDF's text extraction with layout preservation.
    """
    # "text" mode gives clean text; "dict" would give positioned text
    text = page.get_text("text")
    return text


def page_has_images(page: fitz.Page) -> bool:
    """Check if a page contains images (might need OCR)."""
    return len(page.get_images()) > 0


def page_has_tables(page: fitz.Page) -> bool:
    """
    Heuristic: check if a page likely contains tables
    by looking for aligned text patterns.
    """
    text = page.get_text("text")
    lines = text.strip().split("\n")

    # Count lines with multiple tab-separated or space-aligned columns
    tabular_lines = 0
    for line in lines:
        # Check for multiple consecutive spaces (column separators)
        if len(line.split("  ")) >= 3 or "\t" in line:
            tabular_lines += 1

    # If >30% of lines look tabular, flag it
    return tabular_lines > len(lines) * 0.3 if lines else False


def extract_text_from_pdf(file_path: str) -> ExtractionResult:
    """
    Extract all digital text from a PDF file.

    Returns an ExtractionResult with page-by-page content,
    metadata, and flags for OCR needs.

    Args:
        file_path: Path to the PDF file.

    Returns:
        ExtractionResult with extracted content and metadata.
    """
    result = ExtractionResult()

    try:
        doc = fitz.open(file_path)
        result.metadata = extract_metadata(doc)
        result.page_count = doc.page_count

        for page_num in range(doc.page_count):
            page = doc[page_num]
            raw_text = extract_page_text(page)
            cleaned = clean_text(raw_text)
            cleaned = remove_header_footer(cleaned)

            has_img = page_has_images(page)
            has_tbl = page_has_tables(page)

            page_content = PageContent(
                page_number=page_num + 1,  # 1-indexed    
                text=cleaned,
                has_images=has_img,
                has_tables=has_tbl,
                char_count=len(cleaned),
                source_type="text",
            )
            result.pages.append(page_content)
            result.total_chars += len(cleaned)

            # If a page has images but very little text, it may need OCR
            if has_img and len(cleaned) < 50:
                result.needs_ocr = True

        result.has_digital_text = result.total_chars > 100
        doc.close()

        logger.info(
            f"Extracted {result.page_count} pages, "
            f"{result.total_chars} chars from {file_path}"
        )

    except Exception as e:
        logger.error(f"Extraction failed for {file_path}: {e}")
        result.error = str(e)

    return result


def extract_text_for_page(file_path: str, page_number: int) -> Optional[str]:
    """Extract text from a specific page (1-indexed)."""
    try:
        doc = fitz.open(file_path)
        if page_number < 1 or page_number > doc.page_count:
            doc.close()
            return None
        page = doc[page_number - 1]
        text = clean_text(extract_page_text(page))
        doc.close()
        return text
    except Exception as e:
        logger.error(f"Failed to extract page {page_number} from {file_path}: {e}")
        return None


def get_page_count(file_path: str) -> int:
    """Get the total number of pages in a PDF."""
    try:
        doc = fitz.open(file_path)
        count = doc.page_count
        doc.close()
        return count
    except Exception as e:
        logger.error(f"Failed to get page count for {file_path}: {e}")
        return 0
