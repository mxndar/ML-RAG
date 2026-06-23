"""
BM25 Tool — keyword search over document chunks using rank_bm25.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

from app.config import settings
from app.schemas import Chunk

logger = logging.getLogger(__name__)

# Module-level state (loaded once)
_bm25_index = None
_chunks: list[Chunk] = []
_chunk_id_map: dict[str, Chunk] = {}


def load_bm25(indices_dir: Path | None = None) -> None:
    """Load the BM25 index and chunks from disk."""
    global _bm25_index, _chunks, _chunk_id_map

    in_dir = indices_dir or settings.indices_dir

    with open(in_dir / "bm25_index.pkl", "rb") as f:
        _bm25_index = pickle.load(f)

    with open(in_dir / "chunks.pkl", "rb") as f:
        _chunks = pickle.load(f)

    _chunk_id_map = {c.metadata.chunk_id: c for c in _chunks}
    logger.info(f"BM25 tool loaded: {len(_chunks)} chunks")


def is_loaded() -> bool:
    """Check if BM25 index is loaded."""
    return _bm25_index is not None and len(_chunks) > 0


async def bm25_search(query: str, top_k: int | None = None) -> list[Chunk]:
    """
    Search the BM25 index with a query string.

    Args:
        query: Search query.
        top_k: Number of top results to return (default from config).

    Returns:
        List of Chunk objects sorted by BM25 score (descending).
    """
    if not is_loaded():
        raise RuntimeError("BM25 index not loaded. Call load_bm25() first.")

    k = top_k or settings.bm25_top_k
    
    import string
    words = query.lower().split()
    tokenized_query = [w.strip(string.punctuation) for w in words if w.strip(string.punctuation)]

    scores = _bm25_index.get_scores(tokenized_query)

    # Get top-K indices
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

    results = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue  # Skip zero-score results
        chunk = _chunks[idx].model_copy()
        chunk.score = float(scores[idx])
        results.append(chunk)

    logger.info(f"BM25 search '{query[:50]}...' → {len(results)} results")
    return results


def get_chunk_by_id(chunk_id: str) -> Chunk | None:
    """Look up a chunk by its ID."""
    return _chunk_id_map.get(chunk_id)


def get_all_chunks() -> list[Chunk]:
    """Return all indexed chunks."""
    return list(_chunks)
