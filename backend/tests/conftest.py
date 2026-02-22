"""
Shared pytest fixtures for DocBott tests.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.database import Base, User


@pytest.fixture
def engine():
    """Create an in-memory SQLite engine."""
    return create_engine("sqlite:///:memory:", echo=False)


@pytest.fixture
def tables(engine):
    """Create all tables."""
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)


@pytest.fixture
def db_session(engine, tables):
    """Provide a transactional database session."""
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def test_user(db_session):
    """Create and return a test user."""
    from app.auth import hash_password

    user = User(
        username="testuser",
        email="test@example.com",
        hashed_password=hash_password("TestPass123!"),
        role="student",
    )
    db_session.add(user)
    db_session.commit()
    return user


@pytest.fixture
def admin_user(db_session):
    """Create and return an admin user."""
    from app.auth import hash_password

    user = User(
        username="admin",
        email="admin@example.com",
        hashed_password=hash_password("AdminPass123!"),
        role="admin",
    )
    db_session.add(user)
    db_session.commit()
    return user
