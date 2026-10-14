"""
Table Extractor Tool — retrieves parsed tables from the PDF by page number
or by keyword search across table content.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.config import settings
from app.ingestion.table_parser import TableStore, load_table_store
from app.schemas import ExtractedTable

logger = logging.getLogger(__name__)

# Module-level state
_store: TableStore | None = None


def load_tables(indices_dir: Path | None = None) -> None:
    """Load table store from disk."""
    global _store
    _store = load_table_store(indices_dir)
    logger.info(f"Table tool loaded: {len(_store.tables)} tables")


def is_loaded() -> bool:
    """Check if table store is loaded."""
    return _store is not None and len(_store.tables) > 0


# Stopwords to strip from keyword queries before matching
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "of", "in", "to",
    "for", "with", "on", "at", "from", "by", "about", "as", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "and", "but", "or", "nor", "not", "so", "yet", "both", "either",
    "neither", "each", "every", "all", "any", "few", "more", "most",
    "other", "some", "such", "no", "only", "own", "same", "than", "too",
    "very", "just", "because", "if", "when", "where", "how", "what",
    "which", "who", "whom", "this", "that", "these", "those", "i", "me",
    "my", "myself", "we", "our", "you", "your", "he", "him", "she", "her",
    "it", "its", "they", "them", "their",
    # Common verbs in user queries
    "show", "give", "find", "get", "extract", "display", "provide",
    "tell", "list", "fetch", "retrieve", "return", "print",
})


def _tokenize_keyword(keyword: str) -> list[str]:
    """Tokenize a keyword query into meaningful content words."""
    import re
    words = re.findall(r"[a-zA-Z0-9]+", keyword.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) > 1]


def _score_table(table: ExtractedTable, query_words: list[str]) -> int:
    """Score a table by how many query words appear in its content."""
    text = " ".join(table.headers).lower() + " " + " ".join(
        " ".join(row) for row in table.rows
    ).lower() + " " + table.document_name.lower()
    return sum(1 for w in query_words if w in text)


async def table_extract(
    pages: list[int] | None = None,
    keyword: str | None = None,
) -> list[ExtractedTable]:
    """
    Extract tables from the document.

    Args:
        pages: Specific page numbers to get tables from.
               If None, search all tables.
        keyword: Optional keyword to filter tables by content.
                 Uses word-level matching (not whole-string substring).

    Returns:
        List of ExtractedTable objects, ranked by relevance if keyword given.
    """
    if _store is None:
        logger.warning("Table store not loaded — returning empty")
        return []

    results: list[ExtractedTable] = []

    if pages is not None:
        # Get tables from specific pages
        for page_num in pages:
            table_indices = _store.page_index.get(page_num, [])
            for idx in table_indices:
                results.append(_store.tables[idx])
    else:
        # Return all tables (or filter by keyword)
        results = list(_store.tables)

    # Filter by keyword using word-level matching
    if keyword and results:
        query_words = _tokenize_keyword(keyword)
        if query_words:
            scored = []
            for table in results:
                score = _score_table(table, query_words)
                if score > 0:
                    scored.append((score, table))
            # Sort by score descending (most relevant first)
            scored.sort(key=lambda x: x[0], reverse=True)
            results = [t for _, t in scored]
        # If no query words survived stopword removal, return all tables

    logger.info(
        f"Table extract (pages={pages}, keyword={keyword}) → {len(results)} tables"
    )
    return results


def get_all_table_pages() -> list[int]:
    """Return page numbers that contain tables."""
    if _store is None:
        return []
    return sorted(_store.page_index.keys())
