"""
Configuration module for DocBott.
Loads environment variables and provides centralized config access.
All secrets come from environment variables - never hardcoded.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ──────────────────────────────────────────────
# Base paths
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"

# Ensure data directories exist
for d in [PDF_DIR, PROCESSED_DIR, EMBEDDINGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────
# Database
# ──────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'docbott.db'}")

# ──────────────────────────────────────────────
# JWT Authentication
# ──────────────────────────────────────────────
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "change-me-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRATION_MINUTES = int(os.getenv("JWT_EXPIRATION_MINUTES", "60"))

# ──────────────────────────────────────────────
# File Upload
# ──────────────────────────────────────────────
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
ALLOWED_EXTENSIONS = {".pdf"}

# ──────────────────────────────────────────────
# OCR
# ──────────────────────────────────────────────
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "tesseract")
OCR_LANGUAGE = os.getenv("OCR_LANGUAGE", "eng")
OCR_DPI = int(os.getenv("OCR_DPI", "300"))

# ──────────────────────────────────────────────
# Embedding & Vector Store
# ──────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "chroma")  # chroma or faiss
CHROMA_PERSIST_DIR = str(EMBEDDINGS_DIR / "chroma_db")

# ──────────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))

# ──────────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────────
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "5"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.3"))
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.7"))

# ──────────────────────────────────────────────
# AI / Summarization
# ──────────────────────────────────────────────
SUMMARIZATION_MODEL = os.getenv("SUMMARIZATION_MODEL", "facebook/bart-large-cnn")
ENABLE_AI_SUMMARY = os.getenv("ENABLE_AI_SUMMARY", "true").lower() == "true"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# ──────────────────────────────────────────────
# Processing Feature Flags
# ──────────────────────────────────────────────
# Set ENABLE_TABLE_EXTRACTION=false to skip Camelot (much faster, use if no tables needed)
ENABLE_TABLE_EXTRACTION = os.getenv("ENABLE_TABLE_EXTRACTION", "false").lower() == "true"
# Set ENABLE_OCR=false to skip PaddleOCR/Tesseract entirely
ENABLE_OCR = os.getenv("ENABLE_OCR", "true").lower() == "true"
# Pages with fewer than this many characters will trigger OCR (if enabled)
MIN_CHARS_FOR_OCR = int(os.getenv("MIN_CHARS_FOR_OCR", "100"))
# Skip Camelot on documents with more than this many pages (too slow)
MAX_PAGES_FOR_TABLE_EXTRACTION = int(os.getenv("MAX_PAGES_FOR_TABLE_EXTRACTION", "20"))
# Prewarm the embedding model on startup (adds ~5s startup, saves ~10s on first upload)
PREWARM_EMBEDDING_MODEL = os.getenv("PREWARM_EMBEDDING_MODEL", "true").lower() == "true"

# ──────────────────────────────────────────────
# Cache
# ──────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "")
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))

# ──────────────────────────────────────────────
# CORS
# ──────────────────────────────────────────────
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
