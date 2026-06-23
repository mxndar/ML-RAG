"""
Semantic Cache — FAISS + bge-small-en-v1.5 embeddings.
This is the ONLY place embeddings are used for similarity matching.
Threshold: 0.95 cosine similarity.
"""

from __future__ import annotations

import logging
import pickle
import time
from pathlib import Path

import faiss
import numpy as np

from app.config import settings
from app.schemas import CacheEntry, Evidence

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    Embedding-based semantic cache using FAISS.

    - Encodes queries with bge-small-en-v1.5
    - Stores normalized embeddings in FAISS IndexFlatIP (cosine similarity)
    - Match threshold: 0.95
    - Freshness check: stale entries return evidence only (generator must re-run)
    """

    def __init__(
        self,
        threshold: float | None = None,
        ttl_hours: int | None = None,
        model=None,
    ):
        self._threshold = threshold or settings.semantic_threshold
        self._ttl_seconds = (ttl_hours or settings.cache_ttl_hours) * 3600
        self._model = model
        self._dimension: int | None = None
        self._index: faiss.IndexFlatIP | None = None
        self._entries: list[CacheEntry] = []
        self._hits = 0
        self._misses = 0

    def _ensure_model(self):
        """Lazily load the embedding model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading semantic cache embedding model: {settings.embedding_model}")
            self._model = SentenceTransformer(settings.embedding_model)

    def _ensure_index(self):
        """Initialize FAISS index if not yet created."""
        if self._index is None:
            self._ensure_model()
            # Get dimension from a dummy encode
            dummy = self._model.encode(["test"], normalize_embeddings=True)
            self._dimension = dummy.shape[1]
            self._index = faiss.IndexFlatIP(self._dimension)
            logger.info(f"FAISS index initialized (dim={self._dimension})")

    def _encode(self, text: str) -> np.ndarray:
        """Encode a text string into a normalized embedding."""
        self._ensure_model()
        embedding = self._model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False,
        )
        return np.array(embedding, dtype=np.float32)

    def search(self, query: str) -> tuple[CacheEntry | None, float, bool]:
        """
        Search the semantic cache for a similar query.

        Returns:
            (entry_or_none, similarity_score, is_fresh)
            - entry is None on cache miss (similarity < threshold)
            - is_fresh is True if entry is within TTL
        """
        self._ensure_index()

        if self._index.ntotal == 0:
            self._misses += 1
            return None, 0.0, False

        query_embedding = self._encode(query)
        scores, indices = self._index.search(query_embedding, 1)

        similarity = float(scores[0][0])
        idx = int(indices[0][0])

        if similarity < self._threshold:
            self._misses += 1
            logger.debug(f"Semantic cache MISS (similarity={similarity:.4f})")
            return None, similarity, False

        entry = self._entries[idx]
        age = time.time() - entry.timestamp
        is_fresh = age <= self._ttl_seconds

        self._hits += 1
        logger.info(
            f"Semantic cache HIT (similarity={similarity:.4f}, "
            f"age={age:.0f}s, fresh={is_fresh})"
        )
        return entry, similarity, is_fresh

    def add(self, query: str, answer: str, evidence: Evidence) -> None:
        """Add a new entry to the semantic cache."""
        self._ensure_index()

        query_embedding = self._encode(query)
        self._index.add(query_embedding)

        entry = CacheEntry(
            answer=answer,
            evidence=evidence,
            timestamp=time.time(),
            query=query,
        )
        self._entries.append(entry)

        logger.debug(f"Semantic cache ADD (total={self._index.ntotal})")

    def save(self, indices_dir: Path | None = None) -> None:
        """Persist FAISS index and entries to disk."""
        out_dir = indices_dir or settings.indices_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        if self._index is not None and self._index.ntotal > 0:
            faiss.write_index(self._index, str(out_dir / "semantic_cache.faiss"))
            with open(out_dir / "semantic_cache_entries.pkl", "wb") as f:
                pickle.dump(self._entries, f)
            logger.info(f"Saved semantic cache ({self._index.ntotal} entries)")

    def load(self, indices_dir: Path | None = None) -> bool:
        """Load FAISS index and entries from disk. Returns True if loaded."""
        in_dir = indices_dir or settings.indices_dir
        faiss_path = in_dir / "semantic_cache.faiss"
        entries_path = in_dir / "semantic_cache_entries.pkl"

        if not faiss_path.exists() or not entries_path.exists():
            return False

        self._index = faiss.read_index(str(faiss_path))
        self._dimension = self._index.d

        with open(entries_path, "rb") as f:
            self._entries = pickle.load(f)

        logger.info(f"Loaded semantic cache ({self._index.ntotal} entries)")
        return True

    def stats(self) -> dict:
        """Return cache statistics."""
        total = self._hits + self._misses
        return {
            "type": "semantic",
            "size": self._index.ntotal if self._index else 0,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "threshold": self._threshold,
        }
