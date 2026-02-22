"""
OCR module - Optical Character Recognition for scanned PDFs.
Uses PaddleOCR as primary engine with Tesseract as fallback.
Converts PDF pages to images and extracts text.
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple

from app.config import TESSERACT_CMD, OCR_LANGUAGE, OCR_DPI
from app.utils import clean_text, get_logger
from app.extraction import PageContent

logger = get_logger(__name__)


def pdf_page_to_image(file_path: str, page_number: int, dpi: int = None) -> Optional[str]:
    """
    Convert a specific PDF page to a PNG image for OCR.
    Uses pdf2image (poppler-based).

    Args:
        file_path: Path to the PDF file.
        page_number: 1-indexed page number.
        dpi: Resolution for conversion.

    Returns:
        Path to the generated image, or None on failure.
    """
    dpi = dpi or OCR_DPI
    try:
        from pdf2image import convert_from_path

        images = convert_from_path(
            file_path,
            first_page=page_number,
            last_page=page_number,
            dpi=dpi,
        )
        if images:
            # Save to a temp path
            img_dir = Path(file_path).parent / "ocr_temp"
            img_dir.mkdir(exist_ok=True)
            img_path = str(img_dir / f"page_{page_number}.png")
            images[0].save(img_path, "PNG")
            return img_path
    except Exception as e:
        logger.error(f"Failed to convert page {page_number} to image: {e}")
    return None


def ocr_with_paddleocr(image_path: str) -> str:
    """
    Extract text from an image using PaddleOCR.
    PaddleOCR provides better accuracy for complex layouts.
    """
    try:
        from paddleocr import PaddleOCR

        # Initialize with English language, disable GPU if not available
        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        result = ocr.ocr(image_path, cls=True)

        if not result or not result[0]:
            return ""

        # Extract text lines and join them
        lines = []
        for line_info in result[0]:
            if line_info and len(line_info) >= 2:
                text = line_info[1][0]  # text content
                confidence = line_info[1][1]  # confidence score
                if confidence > 0.5:  # filter low-confidence detections
                    lines.append(text)

        return "\n".join(lines)

    except ImportError:
        logger.warning("PaddleOCR not installed, falling back to Tesseract")
        return ""
    except Exception as e:
        logger.error(f"PaddleOCR failed: {e}")
        return ""


def ocr_with_tesseract(image_path: str) -> str:
    """
    Extract text from an image using Tesseract OCR.
    Used as a fallback when PaddleOCR is unavailable.
    """
    try:
        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        text = pytesseract.image_to_string(
            image_path,
            lang=OCR_LANGUAGE,
            config="--psm 6"  # Assume uniform block of text
        )
        return text

    except ImportError:
        logger.warning("pytesseract not installed")
        return ""
    except Exception as e:
        logger.error(f"Tesseract OCR failed: {e}")
        return ""


def ocr_page(file_path: str, page_number: int) -> PageContent:
    """
    Perform OCR on a single PDF page.
    Tries PaddleOCR first, falls back to Tesseract.

    Args:
        file_path: Path to the PDF.
        page_number: 1-indexed page number.

    Returns:
        PageContent with OCR-extracted text.
    """
    image_path = pdf_page_to_image(file_path, page_number)

    text = ""
    if image_path:
        # Try PaddleOCR first
        text = ocr_with_paddleocr(image_path)

        # Fallback to Tesseract
        if not text.strip():
            text = ocr_with_tesseract(image_path)

        # Clean up temp image
        try:
            os.remove(image_path)
        except OSError:
            pass

    cleaned = clean_text(text)

    return PageContent(
        page_number=page_number,
        text=cleaned,
        char_count=len(cleaned),
        source_type="ocr",
    )


def ocr_document(
    file_path: str,
    pages_needing_ocr: Optional[List[int]] = None,
    total_pages: int = 0
) -> List[PageContent]:
    """
    Run OCR on specified pages of a PDF document.

    Args:
        file_path: Path to the PDF.
        pages_needing_ocr: List of 1-indexed page numbers to OCR.
            If None, OCR all pages.
        total_pages: Total page count (used when OCR-ing all pages).

    Returns:
        List of PageContent with OCR results.
    """
    if pages_needing_ocr is None:
        # OCR all pages
        if total_pages == 0:
            from app.extraction import get_page_count
            total_pages = get_page_count(file_path)
        pages_needing_ocr = list(range(1, total_pages + 1))

    results = []
    for page_num in pages_needing_ocr:
        logger.info(f"OCR processing page {page_num} of {file_path}")
        page_content = ocr_page(file_path, page_num)
        results.append(page_content)

    logger.info(f"OCR completed: {len(results)} pages processed")
    return results
