"""
Evidence Merger — combines outputs from all tools into a single
Evidence object, deduplicates, and truncates to ≤300 tokens.
"""

from __future__ import annotations

import logging

from app.config import settings
from app.schemas import Chunk, Evidence, ExtractedTable

logger = logging.getLogger(__name__)


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~0.75 tokens per word."""
    return max(1, int(len(text.split()) * 0.75))


def merge_evidence(
    bm25_chunks: list[Chunk] | None = None,
    page_chunks: list[Chunk] | None = None,
    section_chunks: list[Chunk] | None = None,
    tables: list[ExtractedTable] | None = None,
    max_tokens: int | None = None,
) -> Evidence:
    """
    Merge outputs from all tools into a unified Evidence object.

    Priority order (highest first):
      1. Reranked BM25 chunks (most relevant)
      2. Tables (structured data)
      3. Section content
      4. Page content

    Deduplicates by chunk_id. Truncates to max_tokens (default 300).

    Args:
        bm25_chunks: Reranked BM25 results (typically top 2).
        page_chunks: Chunks from page_retriever.
        section_chunks: Chunks from section_retriever.
        tables: Extracted tables.
        max_tokens: Token budget (default from config).

    Returns:
        Merged Evidence with total_tokens ≤ max_tokens.
    """
    limit = max_tokens or settings.max_evidence_tokens

    seen_chunk_ids: set[str] = set()
    merged_chunks: list[Chunk] = []
    merged_tables: list[ExtractedTable] = []
    total_tokens = 0

    # ── Helper: add chunks with dedup + token budget ──
    def add_chunks(chunks: list[Chunk] | None):
        nonlocal total_tokens
        if not chunks:
            return
        for chunk in chunks:
            if chunk.metadata.chunk_id in seen_chunk_ids:
                continue
            chunk_tokens = _estimate_tokens(chunk.text)
            if total_tokens + chunk_tokens > limit:
                # Try to fit a truncated version
                remaining = limit - total_tokens
                if remaining > 20:
                    words = chunk.text.split()
                    truncated_word_count = int(remaining / 0.75)
                    truncated_text = " ".join(words[:truncated_word_count]) + "..."
                    truncated_chunk = chunk.model_copy()
                    truncated_chunk.text = truncated_text
                    merged_chunks.append(truncated_chunk)
                    seen_chunk_ids.add(chunk.metadata.chunk_id)
                    total_tokens = limit
                return  # Budget exhausted
            seen_chunk_ids.add(chunk.metadata.chunk_id)
            merged_chunks.append(chunk)
            total_tokens += chunk_tokens

    def add_tables(tbls: list[ExtractedTable] | None):
        nonlocal total_tokens
        if not tbls:
            return
        for table in tbls:
            table_tokens = _estimate_tokens(table.markdown)
            if total_tokens + table_tokens > limit:
                continue  # Skip tables that don't fit
            merged_tables.append(table)
            total_tokens += table_tokens

    # ── Add in priority order ──
    add_chunks(bm25_chunks)        # Priority 1: reranked BM25
    add_tables(tables)             # Priority 2: tables
    add_chunks(section_chunks)     # Priority 3: section content
    add_chunks(page_chunks)        # Priority 4: page content

    # ── Compute confidence ──
    if merged_chunks or merged_tables:
        # Confidence based on: having data + BM25 scores
        scores = [c.score for c in merged_chunks if c.score > 0]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        data_coverage = min(1.0, (len(merged_chunks) + len(merged_tables)) / 3)
        confidence = 0.5 * min(1.0, avg_score) + 0.5 * data_coverage
    else:
        confidence = 0.0

    evidence = Evidence(
        chunks=merged_chunks,
        tables=merged_tables,
        total_tokens=total_tokens,
        confidence=round(confidence, 3),
    )

    logger.info(
        f"Evidence merged: {len(merged_chunks)} chunks + "
        f"{len(merged_tables)} tables = {total_tokens} tokens "
        f"(confidence={confidence:.3f})"
    )
    return evidence
