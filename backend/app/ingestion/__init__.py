"""
Ingestion module - Handles PDF upload, validation, and storage.
Validates file type, size, and computes hash for deduplication.
"""

import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import UploadFile, HTTPException, status
from sqlalchemy.orm import Session

from app.config import PDF_DIR, MAX_FILE_SIZE_MB, ALLOWED_EXTENSIONS
from app.database import Document, DocumentStatus
from app.utils import compute_file_hash, safe_filename, get_logger

logger = get_logger(__name__)


def validate_pdf(file: UploadFile) -> None:
    """
    Validate uploaded file is a PDF and within size limits.
    Raises HTTPException on validation failure.
    """
    # Check extension
    if file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type '{ext}'. Only PDF files are allowed."
            )

    # Check content type
    if file.content_type and "pdf" not in file.content_type.lower():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File content type must be application/pdf"
        )


async def save_upload(file: UploadFile) -> Path:
    """
    Save an uploaded file to the PDF directory with a unique name.
    Returns the saved file path.
    """
    # Generate unique filename to avoid collisions
    unique_name = f"{uuid.uuid4().hex}_{safe_filename(file.filename or 'document.pdf')}"
    file_path = PDF_DIR / unique_name

    # Save file
    try:
        with open(file_path, "wb") as buffer:
            content = await file.read()

            # Check file size
            size_mb = len(content) / (1024 * 1024)
            if size_mb > MAX_FILE_SIZE_MB:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File size {size_mb:.1f}MB exceeds limit of {MAX_FILE_SIZE_MB}MB"
                )

            buffer.write(content)
        logger.info(f"Saved upload: {unique_name} ({size_mb:.1f}MB)")
        return file_path

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to save upload: {e}")
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save uploaded file"
        )


def check_duplicate(file_path: str, db: Session) -> Optional[Document]:
    """Check if an identical document (by hash) already exists."""
    file_hash = compute_file_hash(file_path)
    existing = db.query(Document).filter(Document.file_hash == file_hash).first()
    return existing


def create_document_record(
    file_path: Path,
    original_name: str,
    owner_id: int,
    db: Session
) -> Document:
    """
    Create a database record for an uploaded document.
    Computes hash for deduplication tracking.
    """
    file_hash = compute_file_hash(str(file_path))
    file_size = file_path.stat().st_size

    doc = Document(
        filename=file_path.name,
        original_name=original_name,
        file_path=str(file_path),
        file_size=file_size,
        file_hash=file_hash,
        status=DocumentStatus.UPLOADED,
        owner_id=owner_id,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    logger.info(f"Created document record: id={doc.id}, name={original_name}")
    return doc


def delete_document_file(doc: Document) -> None:
    """Remove the physical PDF file from storage."""
    try:
        path = Path(doc.file_path)
        if path.exists():
            path.unlink()
            logger.info(f"Deleted file: {path}")
    except Exception as e:
        logger.error(f"Failed to delete file {doc.file_path}: {e}")
