"""
Tests for the cache module.
"""

import pytest
import time
from app.cache import InMemoryCache, get_cache, cache_query


class TestInMemoryCache:
    """Tests for the in-memory cache implementation."""

    def test_set_and_get(self):
        cache = InMemoryCache(ttl=60)
        cache.set("key1", {"answer": "hello"})
        result = cache.get("key1")
        assert result == {"answer": "hello"}

    def test_get_missing_key(self):
        cache = InMemoryCache(ttl=60)
        result = cache.get("nonexistent")
        assert result is None

    def test_ttl_expiration(self):
        cache = InMemoryCache(ttl=1)  # 1 second TTL
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        time.sleep(1.5)
        assert cache.get("key1") is None

    def test_delete(self):
        cache = InMemoryCache(ttl=60)
        cache.set("key1", "value1")
        cache.delete("key1")
        assert cache.get("key1") is None

    def test_clear(self):
        cache = InMemoryCache(ttl=60)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.clear()
        assert cache.get("k1") is None
        assert cache.get("k2") is None

    def test_overwrite(self):
        cache = InMemoryCache(ttl=60)
        cache.set("key1", "old")
        cache.set("key1", "new")
        assert cache.get("key1") == "new"


class TestCacheQuery:
    """Tests for cache key generation."""

    def test_cache_key_deterministic(self):
        key1 = cache_query("What is AI?", [1, 2])
        key2 = cache_query("What is AI?", [1, 2])
        assert key1 == key2

    def test_cache_key_different_queries(self):
        key1 = cache_query("What is AI?", [1])
        key2 = cache_query("What is ML?", [1])
        assert key1 != key2

    def test_cache_key_different_docs(self):
        key1 = cache_query("What is AI?", [1])
        key2 = cache_query("What is AI?", [2])
        assert key1 != key2


class TestGetCache:
    """Tests for cache singleton."""

    def test_returns_cache_instance(self):
        cache = get_cache()
        assert cache is not None
        assert hasattr(cache, "get")
        assert hasattr(cache, "set")
