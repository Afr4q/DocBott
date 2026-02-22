"""
Chat Memory module - Manages conversation history for contextual Q&A.
Stores chat sessions and messages, provides context window for follow-up questions.
"""

from typing import Dict, List, Optional
from datetime import datetime

from sqlalchemy.orm import Session

from app.database import ChatSession, ChatMessage
from app.utils import get_logger

logger = get_logger(__name__)


class ConversationMemory:
    """
    Manages chat history for a user session.
    Provides context from previous messages to improve follow-up answers.
    """

    def __init__(self, db: Session, user_id: int, session_id: Optional[int] = None):
        self.db = db
        self.user_id = user_id
        self.session_id = session_id

    def create_session(self, title: str = "New Chat") -> ChatSession:
        """Create a new chat session."""
        session = ChatSession(
            user_id=self.user_id,
            title=title,
        )
        self.db.add(session)
        self.db.commit()
        self.db.refresh(session)
        self.session_id = session.id
        logger.info(f"Created chat session {session.id} for user {self.user_id}")
        return session

    def add_message(
        self,
        role: str,
        content: str,
        sources: Optional[List[Dict]] = None,
        confidence: Optional[float] = None,
        ai_summary: Optional[str] = None,
        reasoning: Optional[str] = None,
    ) -> ChatMessage:
        """
        Add a message to the current session.

        Args:
            role: "user" or "assistant"
            content: Message content.
            sources: Retrieved sources (for assistant messages).
            confidence: Confidence score.
            ai_summary: AI-generated summary.
            reasoning: Reasoning explanation.
        """
        if not self.session_id:
            session = self.create_session()
            self.session_id = session.id

        message = ChatMessage(
            session_id=self.session_id,
            role=role,
            content=content,
            sources=sources,
            confidence=confidence,
            ai_summary=ai_summary,
            reasoning=reasoning,
        )
        self.db.add(message)
        self.db.commit()
        self.db.refresh(message)
        return message

    def get_history(self, limit: int = 10) -> List[Dict]:
        """
        Get recent conversation history for context.

        Returns list of {role, content, timestamp, sources, confidence, ai_summary, reasoning} dicts.
        """
        if not self.session_id:
            return []

        messages = (
            self.db.query(ChatMessage)
            .filter(ChatMessage.session_id == self.session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(limit)
            .all()
        )

        # Reverse to get chronological order
        messages.reverse()

        return [
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "sources": msg.sources,
                "confidence": msg.confidence,
                "ai_summary": msg.ai_summary,
                "reasoning": msg.reasoning,
                "timestamp": msg.created_at.isoformat() if msg.created_at else None,
            }
            for msg in messages
        ]

    def get_context_string(self, max_chars: int = 2000) -> str:
        """
        Get conversation context as a formatted string.
        Used to provide context for follow-up questions.
        """
        history = self.get_history(limit=6)
        if not history:
            return ""

        parts = []
        total_chars = 0
        for msg in history:
            line = f"{msg['role'].capitalize()}: {msg['content']}"
            if total_chars + len(line) > max_chars:
                break
            parts.append(line)
            total_chars += len(line)

        return "\n".join(parts)

    def get_sessions(self, limit: int = 20) -> List[Dict]:
        """Get all chat sessions for the user."""
        sessions = (
            self.db.query(ChatSession)
            .filter(ChatSession.user_id == self.user_id)
            .order_by(ChatSession.created_at.desc())
            .limit(limit)
            .all()
        )

        return [
            {
                "id": s.id,
                "title": s.title,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "message_count": len(s.messages) if s.messages else 0,
            }
            for s in sessions
        ]

    def delete_session(self, session_id: int) -> bool:
        """Delete a chat session and all its messages."""
        session = (
            self.db.query(ChatSession)
            .filter(ChatSession.id == session_id, ChatSession.user_id == self.user_id)
            .first()
        )
        if session:
            self.db.delete(session)
            self.db.commit()
            logger.info(f"Deleted chat session {session_id}")
            return True
        return False
