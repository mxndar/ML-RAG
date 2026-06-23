"""
Section Retriever Tool — fetch all chunks from a named section/heading.
Supports fuzzy matching for section names.
"""

from __future__ import annotations

import logging
import pickle
import re
from difflib import SequenceMatcher
from pathlib import Path

from app.config import settings
from app.schemas import Chunk

logger = logging.getLogger(__name__)

# Module-level state
_section_index: dict[str, list[str]] = {}
_chunk_map: dict[str, Chunk] = {}


def load_section_index(indices_dir: Path | None = None) -> None:
    """Load section index and chunks from disk."""
    global _section_index, _chunk_map

    in_dir = indices_dir or settings.indices_dir

    with open(in_dir / "section_index.pkl", "rb") as f:
        _section_index = pickle.load(f)

    with open(in_dir / "chunks.pkl", "rb") as f:
        chunks: list[Chunk] = pickle.load(f)

    _chunk_map = {c.metadata.chunk_id: c for c in chunks}
    logger.info(f"Section tool loaded: {len(_section_index)} sections indexed")


def is_loaded() -> bool:
    """Check if section index is loaded."""
    return len(_section_index) > 0


def get_available_sections() -> list[str]:
    """Return all indexed section names."""
    return sorted(_section_index.keys())


def _normalize_section(name: str) -> str:
    """Normalize a section name for matching."""
    name = name.lower().strip()
    name = re.sub(r"\s+", " ", name)
    # Remove common prefixes
    name = re.sub(r"^(chapter|section|appendix)\s+", "", name)
    return name


def _fuzzy_match_section(query_section: str) -> list[str]:
    """
    Find section names that fuzzy-match the query.

    Returns matching section names from the index, sorted by match quality.
    """
    query_norm = _normalize_section(query_section)
    matches: list[tuple[float, str]] = []

    for section_name in _section_index:
        section_norm = _normalize_section(section_name)

        # Exact match after normalization
        if query_norm == section_norm:
            matches.append((1.0, section_name))
            continue

        # Check if query is a substring
        if query_norm in section_norm or section_norm in query_norm:
            matches.append((0.9, section_name))
            continue

        # Check if query matches section number (e.g., "3.2" matches "3.2 Methods")
        if re.match(rf"^{re.escape(query_norm)}\b", section_norm):
            matches.append((0.95, section_name))
            continue

        # Fuzzy similarity
        ratio = SequenceMatcher(None, query_norm, section_norm).ratio()
        if ratio > 0.5:
            matches.append((ratio, section_name))

    matches.sort(key=lambda x: x[0], reverse=True)
    return [name for _, name in matches]


async def section_retrieve(section_name: str) -> list[Chunk]:
    """
    Retrieve all chunks from a section, with fuzzy name matching.

    Args:
        section_name: Section or heading name to search for
                      (e.g., "chapter 5", "3.2", "METHODOLOGY").

    Returns:
        List of Chunk objects from matching sections.
    """
    if not is_loaded():
        raise RuntimeError("Section index not loaded. Call load_section_index() first.")

    # Try fuzzy match
    matched_sections = _fuzzy_match_section(section_name)

    if not matched_sections:
        logger.info(f"Section retrieve '{section_name}' → no matching sections")
        return []

    # Use the best match (or all close matches)
    seen_ids: set[str] = set()
    results: list[Chunk] = []

    for matched in matched_sections[:3]:  # Top 3 matches at most
        chunk_ids = _section_index.get(matched, [])
        for cid in chunk_ids:
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            chunk = _chunk_map.get(cid)
            if chunk:
                results.append(chunk.model_copy())

    logger.info(
        f"Section retrieve '{section_name}' → matched {matched_sections[:3]} "
        f"→ {len(results)} chunks"
    )
    return results
