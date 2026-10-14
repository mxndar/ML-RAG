"""
Reranker — cross-encoder/ms-marco-MiniLM-L-6-v2.
Only reranks BM25 outputs. No LLM calls, zero token usage.
Returns top 2 chunks.
"""

from __future__ import annotations

import logging
import time

from app.config import settings
from app.schemas import Chunk

logger = logging.getLogger(__name__)

# Module-level model (lazy loaded)
_model = None


def _ensure_model():
    """Lazily load the cross-encoder model."""
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        logger.info(f"Loading reranker model: {settings.reranker_model}")
        _model = CrossEncoder(settings.reranker_model)


async def rerank(
    query: str,
    chunks: list[Chunk],
    top_k: int | None = None,
) -> list[Chunk]:
    """
    Rerank BM25 results using a cross-encoder.

    This is the ONLY reranking in the system — no LLM reranking.
    Zero LLM calls, zero token usage.

    Args:
        query: Original user query.
        chunks: BM25 search results to rerank.
        top_k: Number of top results to return (default from config).

    Returns:
        Top-K chunks sorted by cross-encoder relevance score.
    """
    if not chunks:
        return []

    k = top_k or settings.reranker_top_k
    _ensure_model()

    start = time.time()

    # Prepare (query, passage) pairs for the cross-encoder with document context
    pairs = [
        (query, f"[{chunk.metadata.source or chunk.metadata.document_name}]\n{chunk.text}")
        for chunk in chunks
    ]

    # Score all pairs
    scores = _model.predict(pairs)

    # Attach scores and sort
    scored_chunks: list[Chunk] = []
    for chunk, score in zip(chunks, scores):
        reranked = chunk.model_copy()
        reranked.score = float(score)
        scored_chunks.append(reranked)

    scored_chunks.sort(key=lambda c: c.score, reverse=True)

    top_results = scored_chunks[:k]

    duration = (time.time() - start) * 1000
    logger.info(
        f"Reranked {len(chunks)} chunks → top {len(top_results)} "
        f"(scores: {[f'{c.score:.3f}' for c in top_results]}) "
        f"in {duration:.0f}ms"
    )

    return top_results
