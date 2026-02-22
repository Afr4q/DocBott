"""
Feedback module - Handles user ratings and feedback on answers.
Feedback data can be used to improve future retrieval ranking.
"""

from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from app.database import Feedback
from app.utils import get_logger

logger = get_logger(__name__)


def submit_feedback(
    user_id: int,
    message_id: Optional[int],
    rating: int,
    query: str = "",
    comment: str = "",
    helpful: bool = True,
    db: Session = None,
) -> Feedback:
    """
    Submit user feedback on an answer.

    Args:
        user_id: ID of the user providing feedback.
        message_id: ID of the chat message being rated.
        rating: 1-5 star rating.
        query: The original query (for context).
        comment: Optional text feedback.
        helpful: Whether the answer was helpful.
        db: Database session.

    Returns:
        Created Feedback record.
    """
    if rating < 1 or rating > 5:
        raise ValueError("Rating must be between 1 and 5")

    feedback = Feedback(
        user_id=user_id,
        message_id=message_id,
        rating=rating,
        query=query,
        comment=comment,
        helpful=helpful,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    logger.info(f"Feedback submitted: user={user_id}, rating={rating}")
    return feedback


def get_feedback_stats(db: Session) -> Dict:
    """Get aggregate feedback statistics."""
    feedbacks = db.query(Feedback).all()

    if not feedbacks:
        return {"total": 0, "average_rating": 0, "helpful_pct": 0}

    total = len(feedbacks)
    avg_rating = sum(f.rating for f in feedbacks if f.rating) / total
    helpful = sum(1 for f in feedbacks if f.helpful) / total * 100

    return {
        "total": total,
        "average_rating": round(avg_rating, 2),
        "helpful_pct": round(helpful, 1),
        "rating_distribution": {
            str(i): sum(1 for f in feedbacks if f.rating == i)
            for i in range(1, 6)
        },
    }


def get_low_rated_queries(db: Session, threshold: int = 2) -> List[Dict]:
    """Get queries that received low ratings for improvement analysis."""
    feedbacks = (
        db.query(Feedback)
        .filter(Feedback.rating <= threshold)
        .order_by(Feedback.created_at.desc())
        .limit(50)
        .all()
    )

    return [
        {
            "query": f.query,
            "rating": f.rating,
            "comment": f.comment,
            "created_at": f.created_at.isoformat() if f.created_at else None,
        }
        for f in feedbacks
    ]
