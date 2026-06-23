"""
Page Retriever Tool — fetch all chunks from specific page numbers.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path

from app.config import settings
from app.schemas import Chunk

logger = logging.getLogger(__name__)

# Module-level state
_page_index: dict[int, list[str]] = {}
_chunk_map: dict[str, Chunk] = {}


def load_page_index(indices_dir: Path | None = None) -> None:
    """Load page index and chunks from disk."""
    global _page_index, _chunk_map

    in_dir = indices_dir or settings.indices_dir

    with open(in_dir / "page_index.pkl", "rb") as f:
        _page_index = pickle.load(f)

    with open(in_dir / "chunks.pkl", "rb") as f:
        chunks: list[Chunk] = pickle.load(f)

    _chunk_map = {c.metadata.chunk_id: c for c in chunks}
    logger.info(f"Page tool loaded: {len(_page_index)} pages indexed")


def is_loaded() -> bool:
    """Check if page index is loaded."""
    return len(_page_index) > 0


def get_available_pages() -> list[int]:
    """Return sorted list of available page numbers."""
    return sorted(_page_index.keys())


async def page_retrieve(pages: list[int]) -> list[Chunk]:
    """
    Retrieve all chunks from the specified page numbers.

    Args:
        pages: List of page numbers (1-indexed).

    Returns:
        List of Chunk objects from those pages (deduplicated).
    """
    if not is_loaded():
        raise RuntimeError("Page index not loaded. Call load_page_index() first.")

    seen_ids: set[str] = set()
    results: list[Chunk] = []

    for page_num in pages:
        chunk_ids = _page_index.get(page_num, [])
        if not chunk_ids:
            logger.debug(f"No chunks found for page {page_num}")
            continue

        for cid in chunk_ids:
            if cid in seen_ids:
                continue
            seen_ids.add(cid)

            chunk = _chunk_map.get(cid)
            if chunk:
                results.append(chunk.model_copy())

    logger.info(f"Page retrieve pages={pages} → {len(results)} chunks")
    return results
