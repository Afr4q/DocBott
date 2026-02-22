"""
Tests for the authentication module.
"""

import pytest
from app.auth import hash_password, verify_password, create_access_token, decode_token


class TestPasswordHashing:
    """Tests for password hashing and verification."""

    def test_hash_produces_different_output(self):
        password = "test_password_123"
        hashed = hash_password(password)
        assert hashed != password
        assert len(hashed) > 20

    def test_verify_correct_password(self):
        password = "secure_password"
        hashed = hash_password(password)
        assert verify_password(password, hashed) is True

    def test_verify_wrong_password(self):
        hashed = hash_password("correct_password")
        assert verify_password("wrong_password", hashed) is False

    def test_different_hashes_for_same_password(self):
        password = "same_password"
        hash1 = hash_password(password)
        hash2 = hash_password(password)
        # bcrypt uses random salt, so hashes should differ
        assert hash1 != hash2
        # But both should verify
        assert verify_password(password, hash1)
        assert verify_password(password, hash2)


class TestJWTTokens:
    """Tests for JWT token creation and decoding."""

    def test_create_and_decode_token(self):
        data = {"sub": "1", "role": "student"}
        token = create_access_token(data)
        decoded = decode_token(token)
        assert decoded["sub"] == "1"
        assert decoded["role"] == "student"
        assert "exp" in decoded

    def test_token_contains_expiration(self):
        token = create_access_token({"sub": "1"})
        decoded = decode_token(token)
        assert "exp" in decoded

    def test_invalid_token_raises_error(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException):
            decode_token("invalid.token.here")
