"""
DocBott - Main FastAPI Application
Document-Based Intelligent System with RAG.

This is the application entry point. It registers all routers,
configures CORS, initializes the database, and provides health checks.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import CORS_ORIGINS, LOG_LEVEL, PDF_DIR, PREWARM_EMBEDDING_MODEL
from app.database import init_db

# Configure root logger
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    # Startup
    logger.info("🚀 DocBott starting up...")
    init_db()
    logger.info("✅ Database initialized")

    # Pre-warm embedding model so first upload doesn't pay cold-start cost
    if PREWARM_EMBEDDING_MODEL:
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            logger.info("⏳ Pre-warming embedding model...")
            await loop.run_in_executor(None, _load_embedding_model)
            logger.info("✅ Embedding model ready")
        except Exception as e:
            logger.warning(f"Embedding model pre-warm failed (non-fatal): {e}")

    yield
    # Shutdown
    logger.info("👋 DocBott shutting down...")


def _load_embedding_model():
    """Trigger lazy-load of the embedding model singleton."""
    from app.indexing import get_embedding_model
    get_embedding_model()


# ──────────────────────────────────────────────
# App Instance
# ──────────────────────────────────────────────
app = FastAPI(
    title="DocBott",
    description="Document-Based Intelligent System with RAG",
    version="1.0.0",
    lifespan=lifespan,
)

# ──────────────────────────────────────────────
# CORS
# ──────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────
# Import and register routers
# ──────────────────────────────────────────────
from app.routes import auth_router, documents_router, chat_router, feedback_router, bookmarks_router, faq_router, progress_router, preferences_router, admin_router

app.include_router(auth_router, prefix="/api/auth", tags=["Authentication"])
app.include_router(documents_router, prefix="/api/documents", tags=["Documents"])
app.include_router(chat_router, prefix="/api/chat", tags=["Chat & RAG"])
app.include_router(feedback_router, prefix="/api/feedback", tags=["Feedback"])
app.include_router(bookmarks_router, prefix="/api/bookmarks", tags=["Bookmarks"])
app.include_router(faq_router, prefix="/api/faqs", tags=["FAQs"])
app.include_router(progress_router, prefix="/api/progress", tags=["Reading Progress"])
app.include_router(preferences_router, prefix="/api/preferences", tags=["Preferences"])
app.include_router(admin_router, prefix="/api/admin", tags=["Admin"])


# ──────────────────────────────────────────────
# Health Check
# ──────────────────────────────────────────────
@app.get("/api/health", tags=["System"])
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "DocBott",
        "version": "1.0.0",
    }


@app.get("/", tags=["System"])
def root():
    """Root endpoint with API info."""
    return {
        "message": "Welcome to DocBott - Document Intelligence System",
        "docs": "/docs",
        "health": "/api/health",
    }
