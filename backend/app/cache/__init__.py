"""
Cache module - In-memory caching with optional Redis support.
Caches query results and embeddings to reduce computation.
"""

import hashlib
import json
import time
from typing import Any, Optional

from app.config import REDIS_URL, CACHE_TTL_SECONDS
from app.utils import get_logger

logger = get_logger(__name__)


class InMemoryCache:
    """
    Simple in-memory cache with TTL support.
    Used as default when Redis is not configured.
    """

    def __init__(self, ttl: int = None):
        self.ttl = ttl or CACHE_TTL_SECONDS
        self._store: dict = {}
        self._timestamps: dict = {}

    def _make_key(self, key: str) -> str:
        """Create a consistent cache key."""
        return hashlib.md5(key.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get a cached value. Returns None if expired or missing."""
        hashed = self._make_key(key)

        if hashed not in self._store:
            return None

        # Check TTL
        if time.time() - self._timestamps.get(hashed, 0) > self.ttl:
            self.delete(key)
            return None

        return self._store[hashed]

    def set(self, key: str, value: Any, ttl: int = None) -> None:
        """Set a cached value with optional custom TTL."""
        hashed = self._make_key(key)
        self._store[hashed] = value
        self._timestamps[hashed] = time.time()

    def delete(self, key: str) -> None:
        """Remove a cached value."""
        hashed = self._make_key(key)
        self._store.pop(hashed, None)
        self._timestamps.pop(hashed, None)

    def clear(self) -> None:
        """Clear all cached values."""
        self._store.clear()
        self._timestamps.clear()

    def size(self) -> int:
        """Return number of cached items."""
        return len(self._store)


class RedisCache:
    """Redis-backed cache for distributed deployments."""

    def __init__(self, url: str = None, ttl: int = None):
        self.ttl = ttl or CACHE_TTL_SECONDS
        self.client = None

        try:
            import redis
            self.client = redis.from_url(url or REDIS_URL)
            self.client.ping()
            logger.info("Redis cache connected")
        except Exception as e:
            logger.warning(f"Redis not available: {e}. Falling back to in-memory cache.")
            self.client = None

    @property
    def available(self) -> bool:
        return self.client is not None

    def get(self, key: str) -> Optional[Any]:
        if not self.client:
            return None
        try:
            data = self.client.get(key)
            return json.loads(data) if data else None
        except Exception:
            return None

    def set(self, key: str, value: Any, ttl: int = None) -> None:
        if not self.client:
            return
        try:
            self.client.setex(key, ttl or self.ttl, json.dumps(value, default=str))
        except Exception as e:
            logger.error(f"Redis set failed: {e}")

    def delete(self, key: str) -> None:
        if self.client:
            self.client.delete(key)

    def clear(self) -> None:
        if self.client:
            self.client.flushdb()


# ──────────────────────────────────────────────
# Cache Singleton
# ──────────────────────────────────────────────
_cache = None


def get_cache():
    """Get or create the cache singleton. Uses Redis if configured, else in-memory."""
    global _cache
    if _cache is None:
        if REDIS_URL:
            redis_cache = RedisCache(REDIS_URL)
            if redis_cache.available:
                _cache = redis_cache
            else:
                _cache = InMemoryCache()
        else:
            _cache = InMemoryCache()
    return _cache


def cache_query(query: str, document_ids: list = None) -> str:
    """Generate a cache key for a query + document filter."""
    parts = [query]
    if document_ids:
        parts.append(str(sorted(document_ids)))
    return "|".join(parts)
