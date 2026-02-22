"""
Tests for the feedback module.
Uses an in-memory SQLite database.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import Base, User, Document, ChatSession, ChatMessage, Feedback
from app.feedback import submit_feedback, get_feedback_stats, get_low_rated_queries


@pytest.fixture
def db_session():
    """Create an in-memory database session for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Create test user
    user = User(username="testuser", email="test@test.com", hashed_password="hashed", role="student")
    session.add(user)
    session.commit()

    # Create test document
    doc = Document(filename="test.pdf", file_path="/tmp/test.pdf", file_hash="abc123", status="processed")
    session.add(doc)
    session.commit()

    # Create test chat session and message
    chat_session = ChatSession(user_id=user.id)
    session.add(chat_session)
    session.commit()

    msg = ChatMessage(
        session_id=chat_session.id,
        role="assistant",
        content="Test answer",
        query="What is testing?",
        confidence=0.8,
    )
    session.add(msg)
    session.commit()

    yield session
    session.close()


class TestSubmitFeedback:
    """Tests for submitting feedback."""

    def test_submit_valid_feedback(self, db_session):
        msg = db_session.query(ChatMessage).first()
        user = db_session.query(User).first()

        result = submit_feedback(
            db=db_session,
            message_id=msg.id,
            user_id=user.id,
            rating=5,
            helpful=True,
            comment="Great answer!",
        )
        assert result is not None
        assert result.rating == 5
        assert result.helpful is True

    def test_submit_low_rating(self, db_session):
        msg = db_session.query(ChatMessage).first()
        user = db_session.query(User).first()

        result = submit_feedback(
            db=db_session,
            message_id=msg.id,
            user_id=user.id,
            rating=1,
            helpful=False,
            comment="Not helpful",
        )
        assert result.rating == 1
        assert result.helpful is False


class TestGetFeedbackStats:
    """Tests for feedback statistics."""

    def test_stats_with_feedback(self, db_session):
        msg = db_session.query(ChatMessage).first()
        user = db_session.query(User).first()

        # Add multiple feedback entries
        for rating in [3, 4, 5]:
            fb = Feedback(
                message_id=msg.id,
                user_id=user.id,
                rating=rating,
                helpful=(rating >= 4),
            )
            db_session.add(fb)
        db_session.commit()

        stats = get_feedback_stats(db_session)
        assert stats["total_feedback"] >= 3
        assert "average_rating" in stats

    def test_stats_empty(self, db_session):
        stats = get_feedback_stats(db_session)
        assert stats["total_feedback"] == 0


class TestGetLowRatedQueries:
    """Tests for retrieving low-rated queries."""

    def test_no_low_rated(self, db_session):
        msg = db_session.query(ChatMessage).first()
        user = db_session.query(User).first()

        fb = Feedback(message_id=msg.id, user_id=user.id, rating=5, helpful=True)
        db_session.add(fb)
        db_session.commit()

        low_rated = get_low_rated_queries(db_session, threshold=3)
        assert len(low_rated) == 0

    def test_finds_low_rated(self, db_session):
        msg = db_session.query(ChatMessage).first()
        user = db_session.query(User).first()

        fb = Feedback(message_id=msg.id, user_id=user.id, rating=1, helpful=False)
        db_session.add(fb)
        db_session.commit()

        low_rated = get_low_rated_queries(db_session, threshold=3)
        assert len(low_rated) >= 1
