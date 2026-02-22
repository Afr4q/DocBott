"""
API Routes - All FastAPI route definitions.
Organized into authentication, documents, chat, and feedback routers.
"""

from typing import Dict, List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.database import get_db, User, Document, DocumentStatus, DocumentChunk, UserRole, Bookmark, DocumentTag, ChatSession, ChatMessage, FAQ, ReadingProgress, UserPreference, Feedback
from app.auth import (
    UserCreate, UserLogin, UserResponse, TokenResponse,
    register_user, login_user, get_current_user, require_role,
    hash_password,
)
from app.ingestion import validate_pdf, save_upload, create_document_record
from app.pipeline import process_document, get_full_text
from app.chunking import chunk_document
from app.indexing import index_chunks
from app.rag import query_rag, compare_across_documents
from app.ai import (
    summarize_for_comparison, summarize_text, direct_ai_answer,
    detect_language, translate_text, fact_check_answer,
    generate_faqs_from_text, analyze_document_relationships,
    explain_confidence,
)
from app.memory import ConversationMemory
from app.cache import get_cache, cache_query
from app.feedback import submit_feedback, get_feedback_stats
from app.evaluation import evaluate_answer
from app.utils import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════
# REQUEST / RESPONSE SCHEMAS
# ══════════════════════════════════════════════
class ChatRequest(BaseModel):
    query: str
    document_ids: Optional[List[int]] = None
    session_id: Optional[int] = None
    role: Optional[str] = None  # override user role for this query
    enable_ai_summary: bool = True
    enable_rerank: bool = False
    top_k: int = 5
    answer_mode: str = "pdf"  # "pdf" | "ai" | "both"


class CompareRequest(BaseModel):
    query: str
    document_ids: List[int]
    role: Optional[str] = None


class FeedbackRequest(BaseModel):
    message_id: Optional[int] = None
    rating: int  # 1-5
    query: str = ""
    comment: str = ""
    helpful: bool = True


class BookmarkRequest(BaseModel):
    query: str
    answer: str
    confidence: Optional[float] = None
    sources: Optional[List[Dict]] = None
    note: str = ""


class TagRequest(BaseModel):
    tag: str


class TranslateRequest(BaseModel):
    text: str
    target_lang: str = "en"
    source_lang: str = ""


class FactCheckRequest(BaseModel):
    answer: str
    query: str
    sources: Optional[List[Dict]] = None


class FAQGenerateRequest(BaseModel):
    document_ids: List[int]
    num_questions: int = 5


class ReadingProgressUpdate(BaseModel):
    last_page: int
    total_pages: Optional[int] = None
    notes: Optional[str] = None
    completed: Optional[bool] = None


class ExportRequest(BaseModel):
    format: str = "md"  # "md" | "txt" | "json" | "html"
    session_id: Optional[int] = None
    messages: Optional[List[Dict]] = None


class PreferenceRequest(BaseModel):
    theme: Optional[str] = None
    language: Optional[str] = None
    preferences: Optional[Dict] = None


class HighlightRequest(BaseModel):
    document_id: int
    query: str
    page_number: Optional[int] = None


class DocumentResponse(BaseModel):
    id: int
    filename: str
    original_name: str
    status: str
    page_count: Optional[int] = None
    file_size: Optional[int] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ══════════════════════════════════════════════
# AUTH ROUTER
# ══════════════════════════════════════════════
auth_router = APIRouter()


@auth_router.post("/register", response_model=UserResponse)
def register(user_data: UserCreate, db: Session = Depends(get_db)):
    """Register a new user."""
    user = register_user(user_data, db)
    return UserResponse.model_validate(user)


@auth_router.post("/login", response_model=TokenResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Login and receive JWT token."""
    login_data = UserLogin(username=form_data.username, password=form_data.password)
    return login_user(login_data, db)


@auth_router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    """Get current user profile."""
    return UserResponse.model_validate(current_user)


# ══════════════════════════════════════════════
# DOCUMENTS ROUTER
# ══════════════════════════════════════════════
documents_router = APIRouter()


@documents_router.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Upload a PDF document for processing.
    Returns immediately, processing happens in the background.
    """
    # Validate the file
    validate_pdf(file)

    # Save to disk
    file_path = await save_upload(file)

    # Create database record
    doc = create_document_record(
        file_path=file_path,
        original_name=file.filename or "document.pdf",
        owner_id=current_user.id,
        db=db,
    )

    # Process in the background
    background_tasks.add_task(
        _process_document_task, doc.id, str(file_path)
    )

    return {
        "message": "Document uploaded successfully. Processing started.",
        "document_id": doc.id,
        "filename": doc.original_name,
        "status": doc.status.value,
    }


def _process_document_task(document_id: int, file_path: str):
    """Background task for document processing."""
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        # Step 1: Process (extract text, OCR, tables)
        processed = process_document(file_path, document_id, db)

        if processed.errors and not processed.pages:
            logger.error(f"Processing failed for doc {document_id}: {processed.errors}")
            return

        # Step 2: Chunk the processed content
        pages_data = [
            {
                "page_number": p.page_number,
                "merged_text": p.merged_text,
                "source_types": p.source_types,
            }
            for p in processed.pages
        ]
        chunks = chunk_document(pages_data, document_id)

        # Step 3: Store chunks in DB
        for chunk in chunks:
            db_chunk = DocumentChunk(
                document_id=document_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                page_number=chunk.page_number,
                source_type=chunk.source_type,
                char_count=chunk.char_count,
                embedding_id=chunk.chunk_id,
                metadata_json=chunk.metadata,
            )
            db.add(db_chunk)
        db.commit()

        # Step 4: Index chunks in vector store
        index_chunks(chunks)

        logger.info(f"Document {document_id} fully processed: {len(chunks)} chunks indexed")

    except Exception as e:
        logger.error(f"Background processing failed for doc {document_id}: {e}")
        doc = db.query(Document).filter(Document.id == document_id).first()
        if doc:
            doc.status = DocumentStatus.FAILED
            doc.error_message = str(e)
            db.commit()
    finally:
        db.close()


@documents_router.get("/", response_model=List[DocumentResponse])
def list_documents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all documents for the current user."""
    docs = db.query(Document).filter(Document.owner_id == current_user.id).all()
    return [DocumentResponse.model_validate(d) for d in docs]


@documents_router.get("/{document_id:int}")
def get_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get document details and metadata."""
    doc = db.query(Document).filter(
        Document.id == document_id,
        Document.owner_id == current_user.id,
    ).first()

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    chunk_count = db.query(DocumentChunk).filter(
        DocumentChunk.document_id == document_id
    ).count()

    return {
        "id": doc.id,
        "filename": doc.original_name,
        "status": doc.status.value,
        "page_count": doc.page_count,
        "file_size": doc.file_size,
        "file_hash": doc.file_hash,
        "chunk_count": chunk_count,
        "metadata": doc.metadata_json,
        "created_at": doc.created_at.isoformat() if doc.created_at else None,
        "updated_at": doc.updated_at.isoformat() if doc.updated_at else None,
    }


@documents_router.delete("/{document_id:int}")
def delete_document(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a document and all associated data."""
    doc = db.query(Document).filter(
        Document.id == document_id,
        Document.owner_id == current_user.id,
    ).first()

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Delete from vector store
    try:
        from app.indexing import get_vector_store
        store = get_vector_store()
        if hasattr(store, "delete_by_document"):
            store.delete_by_document(document_id)
    except Exception as e:
        logger.warning(f"Vector store cleanup failed: {e}")

    # Delete from DB (cascade deletes chunks)
    db.delete(doc)
    db.commit()

    return {"message": f"Document {document_id} deleted successfully"}


@documents_router.get("/search/content")
def search_documents_content(
    q: str = Query(..., min_length=1, description="Search query"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Full-text search across all document chunks."""
    # Find user's documents
    user_doc_ids = [
        d.id for d in
        db.query(Document.id).filter(Document.owner_id == current_user.id).all()
    ]

    if not user_doc_ids:
        return {"results": [], "total": 0}

    # Search chunks
    chunks = db.query(DocumentChunk).filter(
        DocumentChunk.document_id.in_(user_doc_ids),
        DocumentChunk.content.ilike(f"%{q}%"),
    ).limit(20).all()

    results = []
    for chunk in chunks:
        doc = db.query(Document).filter(Document.id == chunk.document_id).first()
        # Highlight matching text
        content = chunk.content
        idx = content.lower().find(q.lower())
        start = max(0, idx - 80)
        end = min(len(content), idx + len(q) + 80)
        snippet = ("..." if start > 0 else "") + content[start:end] + ("..." if end < len(content) else "")

        results.append({
            "document_id": chunk.document_id,
            "filename": doc.original_name if doc else "Unknown",
            "page_number": chunk.page_number,
            "snippet": snippet,
            "chunk_id": chunk.id,
        })

    return {"results": results, "total": len(results), "query": q}


# ══════════════════════════════════════════════
# CHAT ROUTER
# ══════════════════════════════════════════════
chat_router = APIRouter()


@chat_router.post("/query")
def chat_query(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Ask a question about uploaded documents.
    Returns grounded answer with sources, confidence, and optional AI summary.
    answer_mode: "pdf" (default), "ai" (direct AI), "both" (PDF + AI side by side)
    """
    # Determine role for answer formatting
    role = request.role or current_user.role.value

    # Check cache
    cache = get_cache()
    cache_key = cache_query(request.query, request.document_ids)
    cached = cache.get(cache_key)
    if cached:
        logger.info("Cache hit for query")
        return cached

    # Build filename mapping
    filenames = {}
    if request.document_ids:
        docs = db.query(Document).filter(Document.id.in_(request.document_ids)).all()
        filenames = {d.id: d.original_name for d in docs}
    else:
        # Use all user's documents
        docs = db.query(Document).filter(
            Document.owner_id == current_user.id,
            Document.status == DocumentStatus.PROCESSED,
        ).all()
        filenames = {d.id: d.original_name for d in docs}

    # ── PDF-based answer (RAG) ──
    rag_response = None
    pdf_answer = ""
    ai_summary = ""
    comparison = {}
    sources = []
    confidence = 0.0
    reasoning = ""
    suggested_questions = []

    if request.answer_mode in ("pdf", "both"):
        rag_response = query_rag(
            query=request.query,
            role=role,
            document_ids=request.document_ids or list(filenames.keys()),
            top_k=request.top_k,
            enable_rerank=request.enable_rerank,
            filenames=filenames,
        )
        pdf_answer = rag_response.answer
        sources = rag_response.sources
        confidence = rag_response.confidence
        reasoning = rag_response.reasoning
        suggested_questions = rag_response.suggested_questions

        # AI summary of PDF content (optional)
        if request.enable_ai_summary and rag_response.sources:
            source_texts = [s.get("content_preview", "") for s in rag_response.sources]
            context = " ".join(source_texts)
            comparison = summarize_for_comparison(
                extractive_answer=rag_response.answer,
                context=context,
                query=request.query,
            )
            ai_summary = comparison.get("ai_answer", "")

    # ── Direct AI answer ──
    direct_ai = ""
    if request.answer_mode in ("ai", "both"):
        # Get context from PDF sources if available
        ai_context = ""
        if rag_response and rag_response.sources:
            ai_context = " ".join(s.get("content_preview", "") for s in rag_response.sources)
        direct_ai = direct_ai_answer(request.query, context=ai_context)

    # Determine the primary answer based on mode
    if request.answer_mode == "ai":
        primary_answer = direct_ai
        reasoning = "Answer generated directly by AI without strict PDF grounding."
    elif request.answer_mode == "both":
        primary_answer = pdf_answer
    else:
        primary_answer = pdf_answer

    # Save to chat memory
    memory = ConversationMemory(db, current_user.id, request.session_id)
    if not request.session_id:
        session = memory.create_session(title=request.query[:100])
    memory.add_message("user", request.query)
    memory.add_message(
        "assistant",
        primary_answer,
        sources=sources,
        confidence=confidence,
        ai_summary=ai_summary or direct_ai,
        reasoning=reasoning,
    )

    # Build response
    response = {
        "answer": primary_answer,
        "ai_summary": ai_summary,
        "direct_ai_answer": direct_ai if request.answer_mode in ("ai", "both") else None,
        "confidence": confidence,
        "sources": sources,
        "reasoning": reasoning,
        "role": role,
        "session_id": memory.session_id,
        "answer_mode": request.answer_mode,
        "suggested_questions": suggested_questions,
        "comparison": comparison if comparison.get("comparison_available") else None,
    }

    # Cache the response
    cache.set(cache_key, response)

    return response


@chat_router.post("/compare")
def compare_documents(
    request: CompareRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Compare answers across multiple documents."""
    if len(request.document_ids) < 2:
        raise HTTPException(status_code=400, detail="At least 2 documents required for comparison")

    # Build filename mapping
    docs = db.query(Document).filter(Document.id.in_(request.document_ids)).all()
    filenames = {d.id: d.original_name for d in docs}

    role = request.role or current_user.role.value

    result = compare_across_documents(
        query=request.query,
        document_ids=request.document_ids,
        role=role,
        filenames=filenames,
    )

    return result


class RelatedRequest(BaseModel):
    query: str
    document_ids: Optional[List[int]] = None
    answer_text: str = ""
    top_k: int = 5


@chat_router.post("/related")
def get_related_content(
    request: RelatedRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Find related/additional content from PDFs beyond the primary answer.
    Uses the original query + answer text to find supplementary information.
    """
    from app.retrieval import hybrid_search
    from app.utils import truncate_text, split_into_sentences

    # Build filename mapping
    filenames = {}
    if request.document_ids:
        docs = db.query(Document).filter(Document.id.in_(request.document_ids)).all()
        filenames = {d.id: d.original_name for d in docs}
    else:
        docs = db.query(Document).filter(
            Document.owner_id == current_user.id,
            Document.status == DocumentStatus.PROCESSED,
        ).all()
        filenames = {d.id: d.original_name for d in docs}

    # Search with a combined query for broader results
    search_query = request.query
    if request.answer_text:
        # Extract key terms from the answer to find related but different content
        keywords = set(request.answer_text.lower().split())
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'can', 'shall',
                      'of', 'in', 'to', 'for', 'with', 'on', 'at', 'by', 'from',
                      'and', 'or', 'but', 'not', 'this', 'that', 'it', 'its'}
        key_terms = [w for w in keywords if w not in stopwords and len(w) > 3][:8]
        if key_terms:
            search_query = f"{request.query} {' '.join(key_terms)}"

    results = hybrid_search(
        search_query,
        document_ids=request.document_ids or list(filenames.keys()),
        top_k=request.top_k + 5,  # fetch extra to filter
    )

    if not results:
        return {"related": [], "total": 0}

    # Filter out chunks that are too similar to the original answer
    answer_lower = request.answer_text.lower() if request.answer_text else ""
    related = []

    for r in results:
        content_lower = r.content.lower()
        # Skip if this chunk is essentially the same as the answer
        if answer_lower and (
            content_lower in answer_lower or
            answer_lower in content_lower
        ):
            continue

        # Extract a useful preview
        sentences = split_into_sentences(r.content)
        preview = " ".join(sentences[:3]) if sentences else r.content[:300]

        related.append({
            "content": preview,
            "page": r.page_number,
            "file": filenames.get(r.document_id, f"Document {r.document_id}"),
            "score": r.final_score,
            "source_type": r.source_type,
        })

        if len(related) >= request.top_k:
            break

    return {"related": related, "total": len(related)}


@chat_router.get("/sessions")
def list_sessions(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all chat sessions for the current user."""
    memory = ConversationMemory(db, current_user.id)
    return {"sessions": memory.get_sessions()}


@chat_router.get("/sessions/{session_id}/history")
def get_session_history(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get message history for a chat session."""
    memory = ConversationMemory(db, current_user.id, session_id)
    return {"messages": memory.get_history(limit=50)}


@chat_router.delete("/sessions/{session_id}")
def delete_session(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a chat session."""
    memory = ConversationMemory(db, current_user.id)
    if memory.delete_session(session_id):
        return {"message": "Session deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


# ══════════════════════════════════════════════
# FEEDBACK ROUTER
# ══════════════════════════════════════════════
feedback_router = APIRouter()


@feedback_router.post("/")
def submit_user_feedback(
    request: FeedbackRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Submit feedback on an answer."""
    feedback = submit_feedback(
        user_id=current_user.id,
        message_id=request.message_id,
        rating=request.rating,
        query=request.query,
        comment=request.comment,
        helpful=request.helpful,
        db=db,
    )
    return {"message": "Feedback submitted", "id": feedback.id}


@feedback_router.get("/stats")
def feedback_statistics(
    current_user: User = Depends(require_role(UserRole.ADMIN, UserRole.TEACHER)),
    db: Session = Depends(get_db),
):
    """Get feedback statistics (admin/teacher only)."""
    return get_feedback_stats(db)


# ══════════════════════════════════════════════
# BOOKMARKS ROUTER
# ══════════════════════════════════════════════
bookmarks_router = APIRouter()


@bookmarks_router.get("/")
def list_bookmarks(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all bookmarks for the current user."""
    bookmarks = db.query(Bookmark).filter(
        Bookmark.user_id == current_user.id
    ).order_by(Bookmark.created_at.desc()).all()

    return [{
        "id": b.id,
        "query": b.query,
        "answer": b.answer,
        "confidence": b.confidence,
        "sources": b.sources,
        "note": b.note,
        "created_at": b.created_at.isoformat() if b.created_at else None,
    } for b in bookmarks]


@bookmarks_router.post("/")
def create_bookmark(
    request: BookmarkRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Save a Q&A pair as a bookmark."""
    bookmark = Bookmark(
        user_id=current_user.id,
        query=request.query,
        answer=request.answer,
        confidence=request.confidence,
        sources=request.sources,
        note=request.note,
    )
    db.add(bookmark)
    db.commit()
    db.refresh(bookmark)
    return {"message": "Bookmark saved", "id": bookmark.id}


@bookmarks_router.delete("/{bookmark_id}")
def delete_bookmark(
    bookmark_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete a bookmark."""
    bookmark = db.query(Bookmark).filter(
        Bookmark.id == bookmark_id,
        Bookmark.user_id == current_user.id,
    ).first()
    if not bookmark:
        raise HTTPException(status_code=404, detail="Bookmark not found")
    db.delete(bookmark)
    db.commit()
    return {"message": "Bookmark deleted"}


# ══════════════════════════════════════════════
# TAGS ROUTER (on documents)
# ══════════════════════════════════════════════

@documents_router.get("/{document_id:int}/tags")
def get_document_tags(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get tags for a document."""
    tags = db.query(DocumentTag).filter(
        DocumentTag.document_id == document_id
    ).all()
    return [{"id": t.id, "tag": t.tag} for t in tags]


@documents_router.post("/{document_id:int}/tags")
def add_document_tag(
    document_id: int,
    request: TagRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Add a tag to a document."""
    # Verify doc ownership
    doc = db.query(Document).filter(
        Document.id == document_id,
        Document.owner_id == current_user.id,
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Check if tag already exists on this document
    existing = db.query(DocumentTag).filter(
        DocumentTag.document_id == document_id,
        DocumentTag.tag == request.tag.strip().lower(),
    ).first()
    if existing:
        return {"message": "Tag already exists", "id": existing.id}

    tag = DocumentTag(
        document_id=document_id,
        tag=request.tag.strip().lower(),
    )
    db.add(tag)
    db.commit()
    return {"message": "Tag added", "id": tag.id, "tag": tag.tag}


@documents_router.delete("/{document_id:int}/tags/{tag_id:int}")
def remove_document_tag(
    document_id: int,
    tag_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Remove a tag from a document."""
    tag = db.query(DocumentTag).filter(
        DocumentTag.id == tag_id,
        DocumentTag.document_id == document_id,
    ).first()
    if not tag:
        raise HTTPException(status_code=404, detail="Tag not found")
    db.delete(tag)
    db.commit()
    return {"message": "Tag removed"}


# ══════════════════════════════════════════════
# ANALYTICS ENDPOINT
# ══════════════════════════════════════════════

@documents_router.get("/analytics/overview")
def documents_analytics(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get document analytics and statistics."""
    from sqlalchemy import func

    user_docs = db.query(Document).filter(Document.owner_id == current_user.id).all()
    doc_ids = [d.id for d in user_docs]

    total_chunks = db.query(func.count(DocumentChunk.id)).filter(
        DocumentChunk.document_id.in_(doc_ids)
    ).scalar() or 0

    total_chars = db.query(func.sum(DocumentChunk.char_count)).filter(
        DocumentChunk.document_id.in_(doc_ids)
    ).scalar() or 0

    total_pages = sum(d.page_count or 0 for d in user_docs)

    total_sessions = db.query(func.count(ChatSession.id)).filter(
        ChatSession.user_id == current_user.id
    ).scalar() or 0

    total_queries = db.query(func.count(ChatMessage.id)).join(
        ChatSession, ChatMessage.session_id == ChatSession.id
    ).filter(
        ChatSession.user_id == current_user.id,
        ChatMessage.role == "user",
    ).scalar() or 0

    # Source type breakdown
    source_breakdown = {}
    src_results = db.query(
        DocumentChunk.source_type, func.count(DocumentChunk.id)
    ).filter(
        DocumentChunk.document_id.in_(doc_ids)
    ).group_by(DocumentChunk.source_type).all()
    for src_type, count in src_results:
        source_breakdown[src_type or "text"] = count

    # Documents by status
    status_counts = {}
    for doc in user_docs:
        s = doc.status.value if hasattr(doc.status, 'value') else str(doc.status)
        status_counts[s] = status_counts.get(s, 0) + 1

    return {
        "total_documents": len(user_docs),
        "total_pages": total_pages,
        "total_chunks": total_chunks,
        "total_characters": total_chars,
        "estimated_words": total_chars // 5 if total_chars else 0,
        "total_sessions": total_sessions,
        "total_queries": total_queries,
        "source_breakdown": source_breakdown,
        "status_counts": status_counts,
    }


# ══════════════════════════════════════════════
# CROSS-LINGUAL ENDPOINTS
# ══════════════════════════════════════════════

@chat_router.post("/translate")
def translate_endpoint(
    request: TranslateRequest,
    current_user: User = Depends(get_current_user),
):
    """Translate text to a target language."""
    source_lang = request.source_lang or detect_language(request.text)
    translated = translate_text(request.text, request.target_lang, source_lang)
    return {
        "original": request.text,
        "translated": translated,
        "source_lang": source_lang,
        "target_lang": request.target_lang,
    }


@chat_router.post("/detect-language")
def detect_language_endpoint(
    request: TranslateRequest,
    current_user: User = Depends(get_current_user),
):
    """Detect the language of provided text."""
    lang = detect_language(request.text)
    lang_names = {
        "en": "English", "es": "Spanish", "fr": "French", "de": "German",
        "hi": "Hindi", "ar": "Arabic", "zh": "Chinese", "ja": "Japanese",
        "ko": "Korean", "ru": "Russian", "pt": "Portuguese", "it": "Italian",
        "th": "Thai", "nl": "Dutch", "sv": "Swedish",
    }
    return {
        "language_code": lang,
        "language_name": lang_names.get(lang, lang),
    }


# ══════════════════════════════════════════════
# FACT-CHECKING ENDPOINTS
# ══════════════════════════════════════════════

@chat_router.post("/fact-check")
def fact_check_endpoint(
    request: FactCheckRequest,
    current_user: User = Depends(get_current_user),
):
    """Fact-check an answer against its sources."""
    result = fact_check_answer(request.answer, request.query, request.sources)
    return result


# ══════════════════════════════════════════════
# CONFIDENCE EXPLAINER
# ══════════════════════════════════════════════

@chat_router.post("/explain-confidence")
def explain_confidence_endpoint(
    request: FactCheckRequest,
    current_user: User = Depends(get_current_user),
):
    """Get a detailed explanation of confidence scoring."""
    # Calculate confidence from sources
    scores = [s.get("score", 0) for s in (request.sources or [])]
    from app.utils import estimate_confidence
    confidence = estimate_confidence(scores)
    result = explain_confidence(confidence, request.sources or [], request.query)
    return result


# ══════════════════════════════════════════════
# FAQ GENERATOR
# ══════════════════════════════════════════════
faq_router = APIRouter()


@faq_router.post("/generate")
def generate_faqs(
    request: FAQGenerateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Auto-generate FAQs from selected documents."""
    all_faqs = []

    for doc_id in request.document_ids:
        doc = db.query(Document).filter(
            Document.id == doc_id,
            Document.owner_id == current_user.id,
        ).first()
        if not doc:
            continue

        # Get document text from chunks
        chunks = db.query(DocumentChunk).filter(
            DocumentChunk.document_id == doc_id
        ).order_by(DocumentChunk.chunk_index).limit(20).all()

        if not chunks:
            continue

        text = " ".join(c.content for c in chunks)
        faqs = generate_faqs_from_text(text, num_questions=request.num_questions)

        page_nums = sorted(set(c.page_number for c in chunks if c.page_number))

        for faq in faqs:
            # Save to DB
            db_faq = FAQ(
                document_id=doc_id,
                question=faq["question"],
                answer=faq["answer"],
                page_numbers=page_nums[:5],
                confidence=0.7,
            )
            db.add(db_faq)
            all_faqs.append({
                "document_id": doc_id,
                "filename": doc.original_name,
                "question": faq["question"],
                "answer": faq["answer"],
                "page_numbers": page_nums[:5],
            })

    db.commit()
    return {"faqs": all_faqs, "total": len(all_faqs)}


@faq_router.get("/")
def list_faqs(
    document_id: Optional[int] = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List FAQs, optionally filtered by document."""
    query = db.query(FAQ)

    if document_id:
        query = query.filter(FAQ.document_id == document_id)
    else:
        # Only return FAQs for user's documents
        user_doc_ids = [
            d.id for d in
            db.query(Document.id).filter(Document.owner_id == current_user.id).all()
        ]
        query = query.filter(FAQ.document_id.in_(user_doc_ids))

    faqs = query.order_by(FAQ.created_at.desc()).all()

    result = []
    for faq in faqs:
        doc = db.query(Document).filter(Document.id == faq.document_id).first()
        result.append({
            "id": faq.id,
            "document_id": faq.document_id,
            "filename": doc.original_name if doc else "Unknown",
            "question": faq.question,
            "answer": faq.answer,
            "page_numbers": faq.page_numbers,
            "confidence": faq.confidence,
            "created_at": faq.created_at.isoformat() if faq.created_at else None,
        })

    return {"faqs": result, "total": len(result)}


@faq_router.delete("/{faq_id}")
def delete_faq(
    faq_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete an FAQ."""
    faq = db.query(FAQ).filter(FAQ.id == faq_id).first()
    if not faq:
        raise HTTPException(status_code=404, detail="FAQ not found")
    # Verify ownership
    doc = db.query(Document).filter(Document.id == faq.document_id, Document.owner_id == current_user.id).first()
    if not doc:
        raise HTTPException(status_code=403, detail="Not authorized")
    db.delete(faq)
    db.commit()
    return {"message": "FAQ deleted"}


# ══════════════════════════════════════════════
# TEXT-TO-SPEECH: EXTRACT TEXT FOR TTS
# ══════════════════════════════════════════════

@documents_router.get("/{document_id:int}/tts")
def get_document_text_for_tts(
    document_id: int,
    page: Optional[int] = None,
    max_chars: int = Query(5000, ge=100, le=50000),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Extract clean text from a document for text-to-speech playback."""
    doc = db.query(Document).filter(
        Document.id == document_id,
        Document.owner_id == current_user.id,
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    query = db.query(DocumentChunk).filter(
        DocumentChunk.document_id == document_id
    ).order_by(DocumentChunk.chunk_index)

    if page is not None:
        query = query.filter(DocumentChunk.page_number == page)

    chunks = query.all()
    if not chunks:
        return {"text": "", "pages": [], "total_chars": 0}

    # Build clean text, track pages
    pages_seen = set()
    segments = []
    total_chars = 0
    for c in chunks:
        if total_chars >= max_chars:
            break
        clean = c.content.strip()
        if clean:
            segments.append(clean)
            total_chars += len(clean)
            if c.page_number:
                pages_seen.add(c.page_number)

    full_text = " ".join(segments)[:max_chars]

    return {
        "document_id": document_id,
        "filename": doc.original_name,
        "text": full_text,
        "total_chars": len(full_text),
        "pages": sorted(pages_seen),
        "page_count": doc.page_count,
    }


# ══════════════════════════════════════════════
# DOCUMENT ANNOTATIONS / NOTES
# ══════════════════════════════════════════════

class AnnotationCreate(BaseModel):
    page_number: int
    content: str
    highlight_text: Optional[str] = ""
    color: Optional[str] = "yellow"

class AnnotationUpdate(BaseModel):
    content: Optional[str] = None
    color: Optional[str] = None

@documents_router.get("/{document_id:int}/annotations")
def get_annotations(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get all annotations for a document."""
    from app.database import Annotation
    doc = db.query(Document).filter(
        Document.id == document_id, Document.owner_id == current_user.id
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    anns = db.query(Annotation).filter(
        Annotation.document_id == document_id,
        Annotation.user_id == current_user.id,
    ).order_by(Annotation.page_number, Annotation.created_at).all()

    return [
        {
            "id": a.id,
            "page_number": a.page_number,
            "content": a.content,
            "highlight_text": a.highlight_text or "",
            "color": a.color or "yellow",
            "created_at": a.created_at.isoformat() if a.created_at else None,
        }
        for a in anns
    ]

@documents_router.post("/{document_id:int}/annotations")
def create_annotation(
    document_id: int,
    request: AnnotationCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Add an annotation to a document page."""
    from app.database import Annotation
    doc = db.query(Document).filter(
        Document.id == document_id, Document.owner_id == current_user.id
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    ann = Annotation(
        document_id=document_id,
        user_id=current_user.id,
        page_number=request.page_number,
        content=request.content,
        highlight_text=request.highlight_text or "",
        color=request.color or "yellow",
    )
    db.add(ann)
    db.commit()
    db.refresh(ann)

    return {
        "id": ann.id,
        "message": "Annotation created",
        "page_number": ann.page_number,
    }

@documents_router.delete("/{document_id:int}/annotations/{annotation_id:int}")
def delete_annotation(
    document_id: int,
    annotation_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Delete an annotation."""
    from app.database import Annotation
    ann = db.query(Annotation).filter(
        Annotation.id == annotation_id,
        Annotation.document_id == document_id,
        Annotation.user_id == current_user.id,
    ).first()
    if not ann:
        raise HTTPException(status_code=404, detail="Annotation not found")

    db.delete(ann)
    db.commit()
    return {"message": "Annotation deleted"}


# ══════════════════════════════════════════════
# STUDY MODE: FLASHCARD GENERATOR
# ══════════════════════════════════════════════

class FlashcardRequest(BaseModel):
    document_id: int
    num_cards: int = 10


def _clean_ocr_text(text: str) -> str:
    """
    Fix common OCR artifacts:
    - Single letters separated by spaces: "D a t a" → "Data"
    - Removes lines that are predominantly single-character tokens (header garbage)
    """
    import re

    # Fix spaced-out characters: sequences of (single-letter SPACE)+ ending in single-letter
    # e.g. "D a t a S t r u c t u r e s" → "DataStructures"
    def _rejoin(m):
        return m.group().replace(" ", "")

    text = re.sub(r'\b(?:[A-Za-z] ){2,}[A-Za-z]\b', _rejoin, text)

    # Remove lines with >50% single-char words (OCR noise lines such as page headers/footers)
    clean_lines = []
    for line in text.splitlines():
        words = line.split()
        if not words:
            continue
        single_char_ratio = sum(1 for w in words if len(w) == 1) / len(words)
        if single_char_ratio > 0.5:
            continue
        clean_lines.append(line)

    return " ".join(clean_lines)


def _is_quality_chunk(text: str) -> bool:
    """Return True if a chunk looks like real readable prose (not OCR garbage)."""
    words = text.split()
    if len(words) < 8:
        return False
    avg_len = sum(len(w) for w in words) / len(words)
    single_char_ratio = sum(1 for w in words if len(w) == 1) / len(words)
    # Good text: avg word length ≥ 4 and very few isolated single letters
    return avg_len >= 4.0 and single_char_ratio < 0.20


@documents_router.post("/flashcards/generate")
def generate_flashcards(
    request: FlashcardRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generate AI-powered flashcards from a document for study mode."""
    doc = db.query(Document).filter(
        Document.id == request.document_id,
        Document.owner_id == current_user.id,
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    chunks = db.query(DocumentChunk).filter(
        DocumentChunk.document_id == request.document_id
    ).order_by(DocumentChunk.chunk_index).all()

    if not chunks:
        return {"flashcards": [], "message": "No content available"}

    # ── 1. Clean OCR artifacts from every chunk, keep only high-quality ones ──
    cleaned_chunks = []
    for c in chunks:
        cleaned = _clean_ocr_text(c.content)
        if _is_quality_chunk(cleaned):
            cleaned_chunks.append(cleaned)

    # If nothing passes quality filter fall back to just cleaning all chunks
    if not cleaned_chunks:
        cleaned_chunks = [_clean_ocr_text(c.content) for c in chunks]

    # Take up to 5000 chars from the best chunks (spread across document)
    step = max(1, len(cleaned_chunks) // 8)
    sampled = cleaned_chunks[::step]
    material = " ".join(sampled)[:5000].strip()

    if not material:
        return {"flashcards": [], "message": "Could not extract readable content"}

    # ── 2. Ask Ollama to generate specific, AI-explained flashcards ──
    prompt = (
        f"You are a university professor creating study flashcards.\n\n"
        f"TASK: Generate exactly {request.num_cards} high-quality flashcards from the study material below.\n\n"
        f"STRICT RULES:\n"
        f"1. 'front': A specific, meaningful question or term (NOT a document title or filename)\n"
        f"2. 'back': A clear, complete explanation written in your own words — expand on the concept for student clarity\n"
        f"3. Focus on: definitions, algorithms, data structure properties, time/space complexity, use cases, comparisons\n"
        f"4. Questions must be self-contained (readable without the source document)\n"
        f"5. Answers should be 2-4 sentences — informative, not just a one-liner\n"
        f"6. NEVER use titles, author names, or book metadata as flashcard content\n"
        f"7. Cover a variety of topics from across the material\n\n"
        f"STUDY MATERIAL:\n---\n{material}\n---\n\n"
        f"Respond ONLY with a valid JSON array of {request.num_cards} objects:\n"
        f'[{{"front": "question here", "back": "AI-explained answer here"}}, ...]'
    )

    from app.ai import query_ollama
    result = query_ollama(
        prompt,
        system_prompt=(
            "You are an expert educator. Generate precise, student-friendly flashcards. "
            "Each answer should be a clear, self-contained explanation — not a raw excerpt. "
            "Respond ONLY in valid JSON array format."
        ),
        max_tokens=3000,
    )

    flashcards = []
    if result:
        try:
            import json, re as _re_fc
            # Try to extract JSON array from the response
            json_match = _re_fc.search(r'\[.*?\]', result, _re_fc.DOTALL)
            if not json_match:
                json_match = _re_fc.search(r'\[.*\]', result, _re_fc.DOTALL)
            if json_match:
                cards = json.loads(json_match.group())
                for card in cards:
                    front = card.get("front", "").strip()
                    back = card.get("back", "").strip()
                    # Skip cards with garbage: too short, spaced chars, or just titles
                    if (
                        front and back
                        and len(front) > 10
                        and len(back) > 20
                        and _is_quality_chunk(back + " " + back)  # back must look like real prose
                    ):
                        flashcards.append({"front": front, "back": back})
        except Exception:
            pass

    # ── 3. Improved sentence-based fallback (only for quality sentences) ──
    if not flashcards:
        import re as _re_fc2
        # Split on sentence boundaries
        all_sentences = _re_fc2.split(r'(?<=[.!?])\s+', material)
        # Keep only real, clean sentences
        good_sentences = [
            s.strip() for s in all_sentences
            if len(s.strip()) > 50
            and _is_quality_chunk(s)
        ]

        for sent in good_sentences:
            if len(flashcards) >= request.num_cards:
                break
            words = sent.split()
            lower = sent.lower()

            # Pattern: "X is/are/refers to Y" → What is X?
            for marker in [' is ', ' are ', ' refers to ', ' means ', ' defined as ']:
                if marker in lower:
                    idx = lower.index(marker)
                    term = sent[:idx].strip()
                    explanation = sent[idx + len(marker):].strip()
                    if len(term) > 4 and len(explanation) > 15 and len(term.split()) <= 6:
                        flashcards.append({
                            "front": f"What is {term}?",
                            "back": f"{term}{marker}{explanation}",
                        })
                        break
            else:
                # Generic: turn a key statement into a question
                if len(words) >= 8:
                    # Build question from 4-5 meaningful words (skip common stopwords)
                    stopwords = {'the', 'a', 'an', 'is', 'are', 'of', 'in', 'to', 'and', 'or'}
                    key_words = [w for w in words[:10] if w.lower() not in stopwords][:5]
                    if key_words:
                        flashcards.append({
                            "front": f"Explain the concept related to: {' '.join(key_words)}",
                            "back": sent,
                        })

    return {
        "document_id": request.document_id,
        "filename": doc.original_name,
        "flashcards": flashcards[:request.num_cards],
        "total": len(flashcards[:request.num_cards]),
    }


# ══════════════════════════════════════════════
# ADMIN: USER MANAGEMENT (Admin-only routes)
# ══════════════════════════════════════════════
admin_router = APIRouter()

@admin_router.get("/users")
def admin_list_users(
    current_user: User = Depends(require_role(UserRole.ADMIN)),
    db: Session = Depends(get_db),
):
    """List all users (admin only)."""
    users = db.query(User).all()
    return [
        {
            "id": u.id,
            "username": u.username,
            "email": u.email,
            "role": u.role.value,
            "is_active": u.is_active,
            "created_at": u.created_at.isoformat() if u.created_at else None,
            "document_count": db.query(Document).filter(Document.owner_id == u.id).count(),
        }
        for u in users
    ]

class RoleUpdateRequest(BaseModel):
    role: str

@admin_router.put("/users/{user_id:int}/role")
def admin_update_role(
    user_id: int,
    request: RoleUpdateRequest,
    current_user: User = Depends(require_role(UserRole.ADMIN)),
    db: Session = Depends(get_db),
):
    """Update a user's role (admin only)."""
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot change your own role")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    try:
        new_role = UserRole(request.role)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid role: {request.role}")

    user.role = new_role
    db.commit()
    return {"message": f"User {user.username} role updated to {new_role.value}"}

@admin_router.put("/users/{user_id:int}/toggle-active")
def admin_toggle_user(
    user_id: int,
    current_user: User = Depends(require_role(UserRole.ADMIN)),
    db: Session = Depends(get_db),
):
    """Activate/deactivate a user (admin only)."""
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot deactivate yourself")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.is_active = not user.is_active
    db.commit()
    return {"message": f"User {user.username} {'activated' if user.is_active else 'deactivated'}"}

@admin_router.get("/stats")
def admin_system_stats(
    current_user: User = Depends(require_role(UserRole.ADMIN)),
    db: Session = Depends(get_db),
):
    """Get system-wide statistics (admin only)."""
    from sqlalchemy import func

    total_users = db.query(User).count()
    active_users = db.query(User).filter(User.is_active == True).count()
    total_docs = db.query(Document).count()
    processed_docs = db.query(Document).filter(Document.status == DocumentStatus.PROCESSED).count()
    failed_docs = db.query(Document).filter(Document.status == DocumentStatus.FAILED).count()
    total_chunks = db.query(DocumentChunk).count()
    total_sessions = db.query(ChatSession).count()
    total_messages = db.query(ChatMessage).count()
    total_faqs = db.query(FAQ).count()
    total_feedbacks = db.query(func.count(Feedback.id)).scalar()

    # Role breakdown
    role_counts = {}
    for role in UserRole:
        role_counts[role.value] = db.query(User).filter(User.role == role).count()

    # Recent activity (last 7 days)
    from datetime import timedelta
    week_ago = datetime.utcnow() - timedelta(days=7)
    recent_docs = db.query(Document).filter(Document.created_at >= week_ago).count()
    recent_users = db.query(User).filter(User.created_at >= week_ago).count()
    recent_messages = db.query(ChatMessage).filter(ChatMessage.created_at >= week_ago).count()

    return {
        "users": {"total": total_users, "active": active_users, "roles": role_counts},
        "documents": {"total": total_docs, "processed": processed_docs, "failed": failed_docs},
        "content": {"total_chunks": total_chunks, "total_faqs": total_faqs},
        "chat": {"total_sessions": total_sessions, "total_messages": total_messages},
        "feedback": {"total": total_feedbacks},
        "recent": {
            "new_documents": recent_docs,
            "new_users": recent_users,
            "new_messages": recent_messages,
        },
    }

@admin_router.delete("/users/{user_id:int}")
def admin_delete_user(
    user_id: int,
    current_user: User = Depends(require_role(UserRole.ADMIN)),
    db: Session = Depends(get_db),
):
    """Delete a user and all their data (admin only)."""
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Delete user's documents (cascade handles chunks)
    docs = db.query(Document).filter(Document.owner_id == user_id).all()
    for doc in docs:
        db.delete(doc)

    db.delete(user)
    db.commit()
    return {"message": f"User {user.username} deleted"}


# ══════════════════════════════════════════════
# DOCUMENT RELATIONSHIP MAPPING
# ══════════════════════════════════════════════

@documents_router.get("/relationships")
def get_document_relationships(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Analyze and return relationships between user's documents."""
    docs = db.query(Document).filter(
        Document.owner_id == current_user.id,
        Document.status == DocumentStatus.PROCESSED,
    ).all()

    if len(docs) < 2:
        return {"nodes": [], "edges": [], "message": "Need at least 2 processed documents"}

    doc_summaries = {}
    for doc in docs:
        chunks = db.query(DocumentChunk).filter(
            DocumentChunk.document_id == doc.id
        ).order_by(DocumentChunk.chunk_index).limit(30).all()
        text_sample = " ".join(c.content for c in chunks)[:3000]
        if len(text_sample.strip()) < 20:
            continue  # Skip docs with no real content
        doc_summaries[doc.id] = {
            "name": doc.original_name,
            "text_sample": text_sample,
            "page_count": doc.page_count,
        }

    if len(doc_summaries) < 2:
        return {"nodes": [], "edges": [], "message": "Need at least 2 documents with sufficient content"}

    result = analyze_document_relationships(doc_summaries)
    return result


# ══════════════════════════════════════════════
# PDF HIGHLIGHTING
# ══════════════════════════════════════════════

@chat_router.post("/highlight")
def highlight_in_pdf(
    request: HighlightRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Find matching text positions in a PDF for highlighting.
    Returns matching chunks with exact positions and page numbers.
    """
    doc = db.query(Document).filter(
        Document.id == request.document_id,
        Document.owner_id == current_user.id,
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Search for matching chunks
    from app.retrieval import hybrid_search
    results = hybrid_search(
        request.query,
        document_ids=[request.document_id],
        top_k=10,
    )

    highlights = []
    for r in results:
        if request.page_number and r.page_number != request.page_number:
            continue

        # Find the best matching snippet within the chunk
        query_words = set(request.query.lower().split())
        sentences = r.content.split(".")
        best_sentence = ""
        best_score = 0
        for sent in sentences:
            overlap = len(query_words & set(sent.lower().split()))
            if overlap > best_score:
                best_score = overlap
                best_sentence = sent.strip()

        highlights.append({
            "page": r.page_number,
            "content": r.content[:500],
            "highlight_text": best_sentence or r.content[:200],
            "score": r.final_score,
            "chunk_index": r.chunk_index if hasattr(r, 'chunk_index') else 0,
            "source_type": r.source_type,
        })

    # Sort by page number
    highlights.sort(key=lambda x: (x["page"], -x["score"]))

    return {
        "document_id": request.document_id,
        "filename": doc.original_name,
        "query": request.query,
        "highlights": highlights[:10],
        "total": len(highlights),
    }


# ══════════════════════════════════════════════
# EXPORT ANSWERS TO MULTIPLE FORMATS
# ══════════════════════════════════════════════

@chat_router.post("/export")
def export_chat(
    request: ExportRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Export chat messages in various formats (md, txt, json, html)."""
    messages = request.messages or []

    # If session_id provided, load from DB
    if request.session_id and not messages:
        memory = ConversationMemory(db, current_user.id, request.session_id)
        messages = memory.get_history(limit=100)

    if not messages:
        raise HTTPException(status_code=400, detail="No messages to export")

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    if request.format == "md":
        content = f"# DocBott Chat Export\n_Exported: {now}_\n\n---\n\n"
        for msg in messages:
            role = msg.get("role", "unknown")
            text = msg.get("content", "")
            if role == "user":
                content += f"**You:** {text}\n\n"
            else:
                content += f"**DocBott:** {text}\n\n"
                conf = msg.get("confidence")
                if conf:
                    content += f"> Confidence: {(conf * 100):.0f}%\n\n"
                sources = msg.get("sources")
                if sources:
                    content += "**Sources:**\n"
                    for s in (sources if isinstance(sources, list) else []):
                        content += f"- {s.get('file', 'Document')}, Page {s.get('page', '?')}\n"
                    content += "\n"
                content += "---\n\n"
        return {"content": content, "format": "md", "filename": f"docbott-chat-{now[:10]}.md"}

    elif request.format == "txt":
        content = f"DocBott Chat Export - {now}\n{'='*50}\n\n"
        for msg in messages:
            role = "You" if msg.get("role") == "user" else "DocBott"
            content += f"{role}: {msg.get('content', '')}\n\n"
        return {"content": content, "format": "txt", "filename": f"docbott-chat-{now[:10]}.txt"}

    elif request.format == "json":
        import json
        export_data = {
            "exported_at": now,
            "service": "DocBott",
            "messages": messages,
        }
        content = json.dumps(export_data, indent=2, default=str)
        return {"content": content, "format": "json", "filename": f"docbott-chat-{now[:10]}.json"}

    elif request.format == "html":
        content = f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>DocBott Chat Export</title>
<style>
body {{ font-family: system-ui; max-width: 800px; margin: 40px auto; padding: 0 20px; background: #f9fafb; }}
.msg {{ margin: 16px 0; padding: 16px; border-radius: 12px; }}
.user {{ background: #4f46e5; color: white; margin-left: 60px; }}
.assistant {{ background: white; border: 1px solid #e5e7eb; margin-right: 60px; }}
h1 {{ color: #1f2937; }} .meta {{ color: #9ca3af; font-size: 12px; margin-top: 8px; }}
</style></head><body>
<h1>DocBott Chat Export</h1><p style="color:#6b7280">Exported: {now}</p><hr>
"""
        for msg in messages:
            cls = "user" if msg.get("role") == "user" else "assistant"
            content += f'<div class="msg {cls}">{msg.get("content", "")}'
            if msg.get("confidence"):
                content += f'<div class="meta">Confidence: {(msg["confidence"]*100):.0f}%</div>'
            content += '</div>\n'
        content += "</body></html>"
        return {"content": content, "format": "html", "filename": f"docbott-chat-{now[:10]}.html"}

    raise HTTPException(status_code=400, detail=f"Unsupported format: {request.format}")


# ══════════════════════════════════════════════
# READING PROGRESS
# ══════════════════════════════════════════════
progress_router = APIRouter()


@progress_router.get("/{document_id:int}")
def get_reading_progress(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get reading progress for a document."""
    progress = db.query(ReadingProgress).filter(
        ReadingProgress.user_id == current_user.id,
        ReadingProgress.document_id == document_id,
    ).first()

    if not progress:
        return {"document_id": document_id, "last_page": 0, "total_pages": 0, "notes": "", "completed": False, "percentage": 0}

    pct = round((progress.last_page / progress.total_pages * 100) if progress.total_pages else 0)
    return {
        "document_id": document_id,
        "last_page": progress.last_page,
        "total_pages": progress.total_pages,
        "notes": progress.notes or "",
        "completed": progress.completed,
        "percentage": pct,
        "updated_at": progress.updated_at.isoformat() if progress.updated_at else None,
    }


@progress_router.put("/{document_id:int}")
def update_reading_progress(
    document_id: int,
    request: ReadingProgressUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update reading progress for a document."""
    progress = db.query(ReadingProgress).filter(
        ReadingProgress.user_id == current_user.id,
        ReadingProgress.document_id == document_id,
    ).first()

    if not progress:
        doc = db.query(Document).filter(Document.id == document_id, Document.owner_id == current_user.id).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        progress = ReadingProgress(
            user_id=current_user.id,
            document_id=document_id,
            total_pages=request.total_pages or doc.page_count or 0,
        )
        db.add(progress)

    progress.last_page = request.last_page
    if request.total_pages is not None:
        progress.total_pages = request.total_pages
    if request.notes is not None:
        progress.notes = request.notes
    if request.completed is not None:
        progress.completed = request.completed

    db.commit()
    pct = round((progress.last_page / progress.total_pages * 100) if progress.total_pages else 0)
    return {"message": "Progress updated", "percentage": pct}


@progress_router.get("/")
def list_reading_progress(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """List all reading progress for user."""
    progress_list = db.query(ReadingProgress).filter(
        ReadingProgress.user_id == current_user.id
    ).all()

    result = []
    for p in progress_list:
        doc = db.query(Document).filter(Document.id == p.document_id).first()
        pct = round((p.last_page / p.total_pages * 100) if p.total_pages else 0)
        result.append({
            "document_id": p.document_id,
            "filename": doc.original_name if doc else "Unknown",
            "last_page": p.last_page,
            "total_pages": p.total_pages,
            "percentage": pct,
            "completed": p.completed,
            "notes": p.notes or "",
        })

    return {"progress": result}


# ══════════════════════════════════════════════
# USER PREFERENCES (Theme, Language)
# ══════════════════════════════════════════════
preferences_router = APIRouter()


@preferences_router.get("/")
def get_preferences(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get user preferences."""
    pref = db.query(UserPreference).filter(
        UserPreference.user_id == current_user.id
    ).first()
    if not pref:
        return {"theme": "light", "language": "en", "preferences": {}}
    return {
        "theme": pref.theme,
        "language": pref.language,
        "preferences": pref.preferences_json or {},
    }


@preferences_router.put("/")
def update_preferences(
    request: PreferenceRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Update user preferences."""
    pref = db.query(UserPreference).filter(
        UserPreference.user_id == current_user.id
    ).first()

    if not pref:
        pref = UserPreference(user_id=current_user.id)
        db.add(pref)

    if request.theme is not None:
        pref.theme = request.theme
    if request.language is not None:
        pref.language = request.language
    if request.preferences is not None:
        pref.preferences_json = request.preferences

    db.commit()
    return {"message": "Preferences updated", "theme": pref.theme, "language": pref.language}


# ══════════════════════════════════════════════
# DOCUMENT INSIGHTS (Smart Summary)
# ══════════════════════════════════════════════

@documents_router.get("/{document_id:int}/insights")
def get_document_insights(
    document_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get smart insights/summary for a document."""
    doc = db.query(Document).filter(
        Document.id == document_id,
        Document.owner_id == current_user.id,
    ).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    chunks = db.query(DocumentChunk).filter(
        DocumentChunk.document_id == document_id
    ).order_by(DocumentChunk.chunk_index).all()

    if not chunks:
        return {"summary": "No content available", "key_topics": [], "stats": {}}

    full_text = " ".join(c.content for c in chunks)

    # Generate summary
    summary = summarize_text(full_text[:3000], max_length=250, min_length=80)

    # Extract key topics via word frequency
    import re as re_mod
    words = re_mod.findall(r'\b[a-zA-Z]{4,}\b', full_text.lower())
    stopwords = {'the', 'and', 'for', 'are', 'was', 'were', 'been', 'have', 'has',
                  'had', 'does', 'did', 'will', 'would', 'could', 'should', 'from',
                  'with', 'this', 'that', 'which', 'their', 'there', 'these', 'those',
                  'about', 'into', 'through', 'during', 'before', 'after', 'above',
                  'also', 'more', 'most', 'other', 'some', 'such', 'than', 'them',
                  'then', 'only', 'very', 'each', 'when', 'what', 'where', 'while',
                  'being', 'between', 'both', 'under', 'over', 'same', 'because'}
    word_freq = {}
    for w in words:
        if w not in stopwords:
            word_freq[w] = word_freq.get(w, 0) + 1
    top_topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]

    # Language detection
    detected_lang = detect_language(full_text[:500])

    # Stats
    total_chars = sum(c.char_count or 0 for c in chunks)
    source_types = set(c.source_type for c in chunks if c.source_type)

    return {
        "document_id": document_id,
        "filename": doc.original_name,
        "summary": summary if summary else "Summary generation unavailable.",
        "key_topics": [{"word": w, "count": c} for w, c in top_topics],
        "detected_language": detected_lang,
        "stats": {
            "total_chunks": len(chunks),
            "total_characters": total_chars,
            "estimated_words": total_chars // 5,
            "page_count": doc.page_count,
            "source_types": list(source_types),
        },
    }


# ══════════════════════════════════════════════
# DOCUMENT COMPARISON
# ══════════════════════════════════════════════

class CompareDocsRequest(BaseModel):
    doc_id_a: int
    doc_id_b: int

@documents_router.post("/compare")
def compare_documents(
    request: CompareDocsRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Compare two documents side-by-side: shared topics, unique topics, stats."""
    import re as re_mod
    from collections import Counter

    doc_a = db.query(Document).filter(Document.id == request.doc_id_a, Document.owner_id == current_user.id).first()
    doc_b = db.query(Document).filter(Document.id == request.doc_id_b, Document.owner_id == current_user.id).first()

    if not doc_a or not doc_b:
        raise HTTPException(status_code=404, detail="One or both documents not found")

    stopwords = {
        'the', 'and', 'for', 'are', 'was', 'were', 'been', 'have', 'has', 'had',
        'does', 'did', 'will', 'would', 'could', 'should', 'from', 'with', 'this',
        'that', 'which', 'their', 'there', 'these', 'those', 'about', 'into',
        'through', 'during', 'before', 'after', 'also', 'more', 'most', 'other',
        'some', 'such', 'than', 'them', 'then', 'only', 'very', 'each', 'when',
        'what', 'where', 'while', 'being', 'between', 'both', 'under', 'over',
        'same', 'because', 'used', 'using', 'like', 'just', 'page', 'figure',
    }

    def get_doc_words(doc_id):
        chunks = db.query(DocumentChunk).filter(DocumentChunk.document_id == doc_id).all()
        text = " ".join(c.content for c in chunks)
        words = re_mod.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        filtered = [w for w in words if w not in stopwords]
        return Counter(filtered), len(words), len(chunks), text

    count_a, wc_a, cc_a, text_a = get_doc_words(request.doc_id_a)
    count_b, wc_b, cc_b, text_b = get_doc_words(request.doc_id_b)

    # Shared words
    shared_keys = set(count_a.keys()) & set(count_b.keys())
    shared_topics = sorted(shared_keys, key=lambda w: count_a[w] + count_b[w], reverse=True)[:15]

    # Unique to each
    unique_a = set(count_a.keys()) - set(count_b.keys())
    unique_b = set(count_b.keys()) - set(count_a.keys())
    top_unique_a = sorted(unique_a, key=lambda w: count_a[w], reverse=True)[:10]
    top_unique_b = sorted(unique_b, key=lambda w: count_b[w], reverse=True)[:10]

    # Jaccard similarity
    all_keys = set(count_a.keys()) | set(count_b.keys())
    jaccard = len(shared_keys) / len(all_keys) if all_keys else 0

    # Reading time (avg 200 words per minute)
    reading_a = max(1, round(wc_a / 200))
    reading_b = max(1, round(wc_b / 200))

    # Language detection
    lang_a = detect_language(text_a[:500])
    lang_b = detect_language(text_b[:500])

    return {
        "doc_a": {
            "id": request.doc_id_a,
            "name": doc_a.original_name,
            "word_count": wc_a,
            "chunk_count": cc_a,
            "page_count": doc_a.page_count,
            "reading_time_min": reading_a,
            "language": lang_a,
            "top_unique_words": [{"word": w, "count": count_a[w]} for w in top_unique_a],
        },
        "doc_b": {
            "id": request.doc_id_b,
            "name": doc_b.original_name,
            "word_count": wc_b,
            "chunk_count": cc_b,
            "page_count": doc_b.page_count,
            "reading_time_min": reading_b,
            "language": lang_b,
            "top_unique_words": [{"word": w, "count": count_b[w]} for w in top_unique_b],
        },
        "shared_topics": [{"word": w, "count_a": count_a[w], "count_b": count_b[w]} for w in shared_topics],
        "similarity": round(jaccard, 3),
        "total_shared_words": len(shared_keys),
    }


# ══════════════════════════════════════════════
# DOCUMENTS OVERVIEW STATS
# ══════════════════════════════════════════════

@documents_router.get("/stats/overview")
def document_stats_overview(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get overview statistics across all user documents."""
    docs = db.query(Document).filter(Document.owner_id == current_user.id).all()
    total_docs = len(docs)
    processed = sum(1 for d in docs if d.status == DocumentStatus.PROCESSED)
    total_pages = sum(d.page_count or 0 for d in docs)

    chunk_count = db.query(DocumentChunk).filter(
        DocumentChunk.document_id.in_([d.id for d in docs])
    ).count() if docs else 0

    total_chars = 0
    if docs:
        from sqlalchemy import func
        result = db.query(func.sum(DocumentChunk.char_count)).filter(
            DocumentChunk.document_id.in_([d.id for d in docs])
        ).scalar()
        total_chars = result or 0

    total_words = total_chars // 5
    reading_time = max(1, round(total_words / 200))

    # FAQ count
    faq_count = 0
    if docs:
        user_doc_ids = [d.id for d in docs]
        faq_count = db.query(FAQ).filter(FAQ.document_id.in_(user_doc_ids)).count()

    return {
        "total_documents": total_docs,
        "processed_documents": processed,
        "total_pages": total_pages,
        "total_chunks": chunk_count,
        "total_words": total_words,
        "estimated_reading_time_min": reading_time,
        "total_faqs": faq_count,
    }
