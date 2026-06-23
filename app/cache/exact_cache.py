"""
Exact Cache — in-memory SHA256 dict with TTL.
Replaces Redis for local-only usage.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time

from app.config import settings
from app.schemas import CacheEntry, Evidence

logger = logging.getLogger(__name__)


class ExactCache:
    """
    In-memory exact-match cache.

    Key:   SHA256(normalized_query)
    Value: CacheEntry(answer, evidence, timestamp)
    TTL:   Configurable (default 24 hours)
    """

    def __init__(self, ttl_hours: int | None = None):
        self._store: dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
        self._ttl_seconds = (ttl_hours or settings.cache_ttl_hours) * 3600
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _make_key(query: str) -> str:
        """Normalize query and compute SHA256 hash."""
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    async def get(self, query: str) -> CacheEntry | None:
        """
        Look up a query in the exact cache.

        Returns None on miss or expired entry.
        """
        key = self._make_key(query)

        async with self._lock:
            entry = self._store.get(key)

            if entry is None:
                self._misses += 1
                return None

            # Check TTL
            age = time.time() - entry.timestamp
            if age > self._ttl_seconds:
                del self._store[key]
                self._misses += 1
                logger.debug(f"Exact cache expired for key {key[:12]}...")
                return None

            self._hits += 1
            logger.info(f"Exact cache HIT (age={age:.0f}s)")
            return entry

    async def set(self, query: str, answer: str, evidence: Evidence) -> None:
        """Store a query result in the cache."""
        key = self._make_key(query)
        entry = CacheEntry(
            answer=answer,
            evidence=evidence,
            timestamp=time.time(),
            query=query,
        )

        async with self._lock:
            self._store[key] = entry

        logger.debug(f"Exact cache SET for key {key[:12]}...")

    async def invalidate(self, query: str) -> None:
        """Remove a specific query from the cache."""
        key = self._make_key(query)
        async with self._lock:
            self._store.pop(key, None)

    async def clear(self) -> None:
        """Clear the entire cache."""
        async with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> dict:
        """Return cache statistics."""
        total = self._hits + self._misses
        return {
            "type": "exact",
            "size": len(self._store),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
        }
