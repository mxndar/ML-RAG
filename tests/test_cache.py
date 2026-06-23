"""Tests for the exact and semantic cache modules."""

import asyncio
import time
import pytest


class TestExactCache:
    def _make_cache(self, ttl_hours=24):
        from app.cache.exact_cache import ExactCache
        return ExactCache(ttl_hours=ttl_hours)

    def _make_evidence(self):
        from app.schemas import Evidence
        return Evidence(chunks=[], tables=[], total_tokens=0, confidence=0.5)

    def test_miss_on_empty(self):
        cache = self._make_cache()
        result = asyncio.run(cache.get("some query"))
        assert result is None

    def test_set_and_get(self):
        cache = self._make_cache()
        evidence = self._make_evidence()
        asyncio.run(cache.set("test query", "test answer", evidence))
        entry = asyncio.run(cache.get("test query"))
        assert entry is not None
        assert entry.answer == "test answer"

    def test_case_insensitive(self):
        cache = self._make_cache()
        evidence = self._make_evidence()
        asyncio.run(cache.set("Test Query", "answer", evidence))
        entry = asyncio.run(cache.get("test query"))
        assert entry is not None

    def test_ttl_expiry(self):
        cache = self._make_cache(ttl_hours=24)
        evidence = self._make_evidence()
        asyncio.run(cache.set("query", "answer", evidence))
        # Manually expire
        key = cache._make_key("query")
        cache._store[key].timestamp = time.time() - (25 * 3600)
        entry = asyncio.run(cache.get("query"))
        assert entry is None

    def test_stats(self):
        cache = self._make_cache()
        evidence = self._make_evidence()
        asyncio.run(cache.get("miss1"))
        asyncio.run(cache.set("hit1", "a", evidence))
        asyncio.run(cache.get("hit1"))
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["size"] == 1


class TestSemanticCache:
    """Basic structural tests (skip if sentence-transformers not available)."""

    def test_stats_empty(self):
        from app.cache.semantic_cache import SemanticCache
        cache = SemanticCache()
        stats = cache.stats()
        assert stats["type"] == "semantic"
        assert stats["hits"] == 0
