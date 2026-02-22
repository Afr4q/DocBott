"""
Pipeline module - Orchestrates the full document processing pipeline.
Merges text from digital extraction, OCR, and tables.
Handles versioning and produces clean, ready-to-chunk content.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from app.config import PROCESSED_DIR, ENABLE_TABLE_EXTRACTION, ENABLE_OCR, MIN_CHARS_FOR_OCR, MAX_PAGES_FOR_TABLE_EXTRACTION
from app.database import Document, DocumentStatus
from app.extraction import extract_text_from_pdf, ExtractionResult, PageContent
from app.ocr import ocr_document
from app.tables import extract_tables_camelot, clean_table_text
from app.utils import clean_text, get_logger

logger = get_logger(__name__)


@dataclass
class ProcessedPage:
    """Final merged content for a single page."""
    page_number: int
    digital_text: str = ""
    ocr_text: str = ""
    table_text: str = ""
    merged_text: str = ""
    source_types: List[str] = field(default_factory=list)


@dataclass
class ProcessedDocument:
    """Complete processed document with all content merged."""
    document_id: int
    filename: str
    pages: List[ProcessedPage] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    page_count: int = 0
    version: int = 1
    total_chars: int = 0
    processing_time: float = 0.0
    errors: List[str] = field(default_factory=list)


def merge_page_content(
    digital: Optional[PageContent],
    ocr: Optional[PageContent],
    table_text: str = ""
) -> ProcessedPage:
    """
    Merge content from different extraction sources for a single page.
    Priority: digital text > OCR text > table text.
    """
    page_num = (digital or ocr).page_number if (digital or ocr) else 0
    page = ProcessedPage(page_number=page_num)

    if digital and digital.text.strip():
        page.digital_text = digital.text
        page.source_types.append("text")

    if ocr and ocr.text.strip():
        page.ocr_text = ocr.text
        page.source_types.append("ocr")

    if table_text.strip():
        page.table_text = table_text
        page.source_types.append("table")

    # Merge strategy: prefer digital text, supplement with OCR for missing parts
    parts = []
    if page.digital_text:
        parts.append(page.digital_text)
    elif page.ocr_text:
        parts.append(page.ocr_text)

    if page.table_text:
        parts.append("\n\n[TABLE]\n" + page.table_text + "\n[/TABLE]")

    page.merged_text = "\n\n".join(parts)
    return page


def process_document(file_path: str, document_id: int, db: Session) -> ProcessedDocument:
    """
    Full processing pipeline for a single document.

    Steps:
    1. Extract digital text (PyMuPDF)
    2. Identify pages needing OCR
    3. Run OCR on those pages
    4. Extract tables
    5. Merge all content
    6. Save processed output
    7. Update document status

    Args:
        file_path: Path to the PDF file.
        document_id: Database ID of the document.
        db: Database session.

    Returns:
        ProcessedDocument with all merged content.
    """
    import time
    start_time = time.time()

    result = ProcessedDocument(document_id=document_id, filename=Path(file_path).name)

    # Update status to processing
    doc = db.query(Document).filter(Document.id == document_id).first()
    if doc:
        doc.status = DocumentStatus.PROCESSING
        db.commit()

    try:
        # Step 1: Digital text extraction
        logger.info(f"Step 1: Extracting digital text from {file_path}")
        extraction = extract_text_from_pdf(file_path)
        result.metadata = extraction.metadata
        result.page_count = extraction.page_count

        if extraction.error:
            result.errors.append(f"Extraction error: {extraction.error}")

        # Step 2: Identify pages needing OCR
        pages_for_ocr = []
        if ENABLE_OCR and extraction.needs_ocr:
            for page in extraction.pages:
                # Only run OCR if page has very little digital text
                if page.char_count < MIN_CHARS_FOR_OCR:
                    pages_for_ocr.append(page.page_number)

        # Step 3: Run OCR on identified pages
        ocr_results = {}
        if pages_for_ocr:
            logger.info(f"Step 2: Running OCR on {len(pages_for_ocr)} pages (of {result.page_count} total)")
            ocr_pages = ocr_document(file_path, pages_for_ocr)
            for ocr_page in ocr_pages:
                ocr_results[ocr_page.page_number] = ocr_page
        else:
            logger.info(f"Step 2: OCR skipped ({'disabled' if not ENABLE_OCR else 'no pages need it'})")

        # Step 4: Extract tables (optional - controlled by ENABLE_TABLE_EXTRACTION)
        tables_by_page: Dict[int, str] = {}
        if ENABLE_TABLE_EXTRACTION and result.page_count <= MAX_PAGES_FOR_TABLE_EXTRACTION:
            logger.info(f"Step 3: Extracting tables from {result.page_count} pages")
            try:
                tables = extract_tables_camelot(file_path)
                for table in tables:
                    cleaned = clean_table_text(table)
                    page_num = cleaned.page_number
                    if page_num not in tables_by_page:
                        tables_by_page[page_num] = ""
                    tables_by_page[page_num] += "\n" + cleaned.to_text()
                logger.info(f"Step 3: Found tables on {len(tables_by_page)} pages")
            except Exception as e:
                logger.warning(f"Table extraction failed: {e}")
                result.errors.append(f"Table extraction: {e}")
        else:
            reason = "disabled" if not ENABLE_TABLE_EXTRACTION else f"doc has {result.page_count} pages (limit: {MAX_PAGES_FOR_TABLE_EXTRACTION})"
            logger.info(f"Step 3: Table extraction skipped ({reason})")

        # Step 5: Merge all content
        logger.info("Step 4: Merging content")
        for page in extraction.pages:
            ocr_page = ocr_results.get(page.page_number)
            table_text = tables_by_page.get(page.page_number, "")

            processed_page = merge_page_content(page, ocr_page, table_text)
            result.pages.append(processed_page)
            result.total_chars += len(processed_page.merged_text)

        # Step 6: Save processed output
        save_processed_output(result)

        # Step 7: Update document status
        if doc:
            doc.status = DocumentStatus.PROCESSED
            doc.page_count = result.page_count
            doc.metadata_json = result.metadata
            db.commit()

        result.processing_time = time.time() - start_time
        logger.info(
            f"Processing complete: {result.page_count} pages, "
            f"{result.total_chars} chars in {result.processing_time:.2f}s"
        )

    except Exception as e:
        logger.error(f"Pipeline failed for document {document_id}: {e}")
        result.errors.append(str(e))
        if doc:
            doc.status = DocumentStatus.FAILED
            doc.error_message = str(e)
            db.commit()

    return result


def save_processed_output(result: ProcessedDocument) -> Path:
    """Save processed document content as JSON for downstream use."""
    output_path = PROCESSED_DIR / f"doc_{result.document_id}_v{result.version}.json"

    output = {
        "document_id": result.document_id,
        "filename": result.filename,
        "page_count": result.page_count,
        "version": result.version,
        "metadata": result.metadata,
        "pages": [
            {
                "page_number": p.page_number,
                "merged_text": p.merged_text,
                "source_types": p.source_types,
            }
            for p in result.pages
        ],
        "processed_at": datetime.utcnow().isoformat(),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved processed output: {output_path}")
    return output_path


def get_full_text(result: ProcessedDocument) -> str:
    """Get the full merged text from all pages."""
    return "\n\n".join(
        f"[Page {p.page_number}]\n{p.merged_text}"
        for p in result.pages
        if p.merged_text.strip()
    )
