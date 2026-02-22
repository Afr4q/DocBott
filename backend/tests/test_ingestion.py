"""
Tests for the ingestion module.
Uses mock file objects - does not require actual file uploads.
"""

import pytest
import os
import tempfile
from unittest.mock import MagicMock, AsyncMock, patch
from app.ingestion import validate_pdf, check_duplicate


class TestValidatePdf:
    """Tests for PDF validation."""

    def test_valid_pdf_filename(self):
        mock_file = MagicMock()
        mock_file.filename = "document.pdf"
        mock_file.size = 1024 * 1024  # 1MB
        mock_file.content_type = "application/pdf"

        is_valid, error = validate_pdf(mock_file)
        assert is_valid is True
        assert error is None

    def test_invalid_extension(self):
        mock_file = MagicMock()
        mock_file.filename = "document.txt"
        mock_file.size = 1024
        mock_file.content_type = "text/plain"

        is_valid, error = validate_pdf(mock_file)
        assert is_valid is False
        assert "pdf" in error.lower()

    def test_file_too_large(self):
        mock_file = MagicMock()
        mock_file.filename = "big.pdf"
        mock_file.size = 500 * 1024 * 1024  # 500MB
        mock_file.content_type = "application/pdf"

        is_valid, error = validate_pdf(mock_file)
        assert is_valid is False
        assert "size" in error.lower() or "large" in error.lower()

    def test_empty_filename(self):
        mock_file = MagicMock()
        mock_file.filename = ""
        mock_file.size = 1024

        is_valid, error = validate_pdf(mock_file)
        assert is_valid is False


class TestCheckDuplicate:
    """Tests for duplicate detection."""

    def test_no_duplicate(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from app.database import Base

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        db = Session()

        result = check_duplicate(db, "unique_hash_123")
        assert result is None

    def test_finds_duplicate(self):
        from sqlalchemy import create_engine
        from sqlalchemy.orm import sessionmaker
        from app.database import Base, Document

        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(engine)
        Session = sessionmaker(bind=engine)
        db = Session()

        doc = Document(
            filename="test.pdf",
            file_path="/tmp/test.pdf",
            file_hash="hash_abc",
            status="processed",
        )
        db.add(doc)
        db.commit()

        result = check_duplicate(db, "hash_abc")
        assert result is not None
        assert result.filename == "test.pdf"
