"""
Table Parser — extracts tables from PDF pages using pdfplumber.
Stores them indexed by page number for the Table Extractor tool.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

from app.config import settings
from app.schemas import ExtractedTable

logger = logging.getLogger(__name__)


@dataclass
class TableStore:
    """All tables extracted from a document, indexed by page number."""
    tables: list[ExtractedTable] = field(default_factory=list)
    page_index: dict[int, list[int]] = field(default_factory=dict)  # page_num → [table list indices]


def extract_tables(pdf_path: str | Path) -> TableStore:
    """
    Extract all tables from a PDF using pdfplumber.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        TableStore with tables indexed by page number.
    """
    import pdfplumber

    path = Path(pdf_path)
    store = TableStore()

    logger.info(f"Extracting tables from '{path.name}'")

    try:
        with pdfplumber.open(str(path)) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                page_num = page_idx + 1
                page_tables = page.extract_tables() or []

                for tbl_idx, table_data in enumerate(page_tables):
                    if not table_data or len(table_data) < 2:
                        continue  # Skip empty or single-row tables

                    # First row as headers (cleaned)
                    headers = [
                        str(cell).strip() if cell else ""
                        for cell in table_data[0]
                    ]

                    # Remaining rows as data
                    rows = []
                    for row in table_data[1:]:
                        cleaned_row = [
                            str(cell).strip() if cell else ""
                            for cell in row
                        ]
                        rows.append(cleaned_row)

                    # Convert to markdown
                    markdown = _table_to_markdown(headers, rows)

                    table = ExtractedTable(
                        page_number=page_num,
                        table_index=tbl_idx,
                        document_name=path.stem,
                        headers=headers,
                        rows=rows,
                        markdown=markdown,
                    )
                    idx = len(store.tables)
                    store.tables.append(table)
                    store.page_index.setdefault(page_num, []).append(idx)

    except Exception as e:
        logger.error(f"Error extracting tables: {e}")

    logger.info(
        f"Extracted {len(store.tables)} tables from "
        f"{len(store.page_index)} pages"
    )
    return store


def _table_to_markdown(headers: list[str], rows: list[list[str]]) -> str:
    """Convert a table to markdown format."""
    if not headers:
        return ""

    # Header row
    md = "| " + " | ".join(headers) + " |\n"
    # Separator
    md += "| " + " | ".join("---" for _ in headers) + " |\n"
    # Data rows
    for row in rows:
        # Pad row if needed
        padded = row + [""] * (len(headers) - len(row))
        md += "| " + " | ".join(padded[: len(headers)]) + " |\n"

    return md


def save_table_store(store: TableStore, indices_dir: Path | None = None) -> None:
    """Persist table store to disk."""
    out_dir = indices_dir or settings.indices_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "tables.pkl", "wb") as f:
        pickle.dump(store, f)

    logger.info(f"Saved {len(store.tables)} tables to {out_dir / 'tables.pkl'}")


def load_table_store(indices_dir: Path | None = None) -> TableStore:
    """Load table store from disk."""
    in_dir = indices_dir or settings.indices_dir
    tables_path = in_dir / "tables.pkl"

    if not tables_path.exists():
        logger.warning("No tables.pkl found — returning empty store")
        return TableStore()

    with open(tables_path, "rb") as f:
        store = pickle.load(f)

    logger.info(f"Loaded {len(store.tables)} tables from {tables_path}")
    return store
