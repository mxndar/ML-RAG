"""
PDF Loader — extracts text page-by-page from PDF files using pypdf.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    """Raw text content from a single PDF page."""
    page_number: int
    text: str
    document_name: str = ""


@dataclass
class LoaderResult:
    """Result from loading a document."""
    pages: list[PageContent] = field(default_factory=list)
    total_pages: int = 0
    document_name: str = ""
    errors: list[str] = field(default_factory=list)


def load_pdf(file_path: str | Path) -> LoaderResult:
    """
    Load a PDF and extract text from every page.

    Args:
        file_path: Path to the PDF file.

    Returns:
        LoaderResult with page-by-page text content.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected .pdf file, got: {path.suffix}")

    doc_name = path.stem
    result = LoaderResult(document_name=doc_name)

    try:
        reader = PdfReader(str(path))
        result.total_pages = len(reader.pages)
        logger.info(f"Loading PDF '{doc_name}' — {result.total_pages} pages")

        for i, page in enumerate(reader.pages):
            page_num = i + 1  # 1-indexed
            try:
                text = page.extract_text() or ""
                result.pages.append(
                    PageContent(
                        page_number=page_num,
                        text=text,
                        document_name=doc_name,
                    )
                )
            except Exception as e:
                error_msg = f"Error extracting page {page_num}: {e}"
                logger.warning(error_msg)
                result.errors.append(error_msg)
                # Still add the page with empty text so page indexing stays aligned
                result.pages.append(
                    PageContent(
                        page_number=page_num,
                        text="",
                        document_name=doc_name,
                    )
                )

        logger.info(
            f"Loaded {len(result.pages)} pages from '{doc_name}' "
            f"({len(result.errors)} errors)"
        )

    except Exception as e:
        raise RuntimeError(f"Failed to load PDF '{path}': {e}") from e

    return result
