"""
Tests for the memory (conversation history) module.
Uses an in-memory SQLite database.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import Base, User, ChatSession, ChatMessage
from app.memory import ConversationMemory


@pytest.fixture
def db_session():
    """Create in-memory DB session."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    user = User(username="testuser", email="t@t.com", hashed_password="h", role="student")
    session.add(user)
    session.commit()

    yield session
    session.close()


class TestConversationMemory:
    """Tests for ConversationMemory class."""

    def test_create_session(self, db_session):
        user = db_session.query(User).first()
        memory = ConversationMemory(db_session)

        session = memory.create_session(user_id=user.id, title="Test Chat")
        assert session is not None
        assert session.user_id == user.id

    def test_add_and_get_history(self, db_session):
        user = db_session.query(User).first()
        memory = ConversationMemory(db_session)

        session = memory.create_session(user_id=user.id)
        memory.add_message(
            session_id=session.id,
            role="user",
            content="What is Python?",
            query="What is Python?",
        )
        memory.add_message(
            session_id=session.id,
            role="assistant",
            content="Python is a programming language.",
            query="What is Python?",
            confidence=0.9,
        )

        history = memory.get_history(session_id=session.id)
        assert len(history) == 2
        assert history[0].role == "user"
        assert history[1].role == "assistant"

    def test_get_context_string(self, db_session):
        user = db_session.query(User).first()
        memory = ConversationMemory(db_session)

        session = memory.create_session(user_id=user.id)
        memory.add_message(session_id=session.id, role="user", content="Hi")
        memory.add_message(session_id=session.id, role="assistant", content="Hello!")

        context = memory.get_context_string(session_id=session.id)
        assert "user:" in context.lower() or "User:" in context
        assert "Hello!" in context

    def test_get_sessions(self, db_session):
        user = db_session.query(User).first()
        memory = ConversationMemory(db_session)

        memory.create_session(user_id=user.id, title="Chat 1")
        memory.create_session(user_id=user.id, title="Chat 2")

        sessions = memory.get_sessions(user_id=user.id)
        assert len(sessions) >= 2

    def test_delete_session(self, db_session):
        user = db_session.query(User).first()
        memory = ConversationMemory(db_session)

        session = memory.create_session(user_id=user.id)
        session_id = session.id

        memory.delete_session(session_id=session_id)
        sessions = memory.get_sessions(user_id=user.id)
        ids = [s.id for s in sessions]
        assert session_id not in ids

    def test_empty_history(self, db_session):
        user = db_session.query(User).first()
        memory = ConversationMemory(db_session)

        session = memory.create_session(user_id=user.id)
        history = memory.get_history(session_id=session.id)
        assert history == []
