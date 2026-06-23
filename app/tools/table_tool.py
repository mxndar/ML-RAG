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

    Returns:
        List of ExtractedTable objects.
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

    # Filter by keyword if provided
    if keyword and results:
        keyword_lower = keyword.lower()
        filtered = []
        for table in results:
            # Search in headers and cell values
            text = " ".join(table.headers) + " " + " ".join(
                " ".join(row) for row in table.rows
            )
            if keyword_lower in text.lower():
                filtered.append(table)
        results = filtered

    logger.info(
        f"Table extract (pages={pages}, keyword={keyword}) → {len(results)} tables"
    )
    return results


def get_all_table_pages() -> list[int]:
    """Return page numbers that contain tables."""
    if _store is None:
        return []
    return sorted(_store.page_index.keys())
