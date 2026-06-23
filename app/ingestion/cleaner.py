"""
Text Cleaner — removes headers, footers, duplicate whitespace.
Explicitly PRESERVES stop words (no stop word removal).
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """
    Clean a single page's text.

    Operations (in order):
      1. Unicode normalize (NFKD → NFC)
      2. Remove null bytes and control characters (keep newlines, tabs)
      3. Collapse multiple spaces on the same line to one
      4. Collapse 3+ newlines to 2
      5. Strip leading/trailing whitespace

    Explicitly does NOT remove stop words.
    """
    if not text:
        return ""

    # 1 — Unicode normalize
    text = unicodedata.normalize("NFKC", text)

    # 2 — Remove control chars except \n \r \t
    text = re.sub(r"[^\S \n\r\t]", " ", text)  # weird whitespace → space
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # 3 — Collapse multiple spaces (not newlines) to single space
    text = re.sub(r"[ \t]+", " ", text)

    # 4 — Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 5 — Strip
    text = text.strip()

    return text


def detect_headers_footers(pages_text: list[str], threshold: float = 0.6) -> tuple[set[str], set[str]]:
    """
    Detect repeated headers/footers by finding lines that appear
    on more than `threshold` fraction of pages.

    Looks at the first 3 lines (headers) and last 3 lines (footers) of each page.

    Returns:
        (header_lines, footer_lines) — sets of normalized line strings to remove.
    """
    if len(pages_text) < 5:
        # Too few pages to reliably detect
        return set(), set()

    n_pages = len(pages_text)
    min_count = int(n_pages * threshold)

    header_counter: Counter[str] = Counter()
    footer_counter: Counter[str] = Counter()

    for page_text in pages_text:
        lines = [ln.strip() for ln in page_text.split("\n") if ln.strip()]
        if not lines:
            continue

        # First 3 lines → potential headers
        for line in lines[:3]:
            normalized = _normalize_for_comparison(line)
            if normalized and len(normalized) > 2:
                header_counter[normalized] += 1

        # Last 3 lines → potential footers
        for line in lines[-3:]:
            normalized = _normalize_for_comparison(line)
            if normalized and len(normalized) > 2:
                footer_counter[normalized] += 1

    headers = {line for line, count in header_counter.items() if count >= min_count}
    footers = {line for line, count in footer_counter.items() if count >= min_count}

    if headers:
        logger.info(f"Detected {len(headers)} header patterns")
    if footers:
        logger.info(f"Detected {len(footers)} footer patterns")

    return headers, footers


def remove_headers_footers(text: str, headers: set[str], footers: set[str]) -> str:
    """Remove detected header/footer lines from page text."""
    if not headers and not footers:
        return text

    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        normalized = _normalize_for_comparison(line.strip())
        if normalized in headers or normalized in footers:
            continue
        cleaned_lines.append(line)

    return "\n".join(cleaned_lines)


def remove_page_numbers(text: str) -> str:
    """Remove standalone page numbers (e.g., '  42  ' or 'Page 42' or '- 42 -')."""
    # Standalone number on its own line
    text = re.sub(r"^\s*\d{1,4}\s*$", "", text, flags=re.MULTILINE)
    # "Page X" or "page X of Y"
    text = re.sub(r"^\s*[Pp]age\s+\d+(\s+of\s+\d+)?\s*$", "", text, flags=re.MULTILINE)
    # "- X -" format
    text = re.sub(r"^\s*[-–—]\s*\d+\s*[-–—]\s*$", "", text, flags=re.MULTILINE)
    return text


def clean_document(pages_text: list[str]) -> list[str]:
    """
    Full cleaning pipeline for a multi-page document.

    1. Detect headers/footers across all pages
    2. For each page: remove headers/footers → remove page numbers → clean text

    Args:
        pages_text: List of raw text strings, one per page.

    Returns:
        List of cleaned text strings, one per page.
    """
    logger.info(f"Cleaning {len(pages_text)} pages")

    # Step 1 — Detect repeated headers/footers
    headers, footers = detect_headers_footers(pages_text)

    # Step 2 — Clean each page
    cleaned = []
    for page_text in pages_text:
        text = remove_headers_footers(page_text, headers, footers)
        text = remove_page_numbers(text)
        text = clean_text(text)
        cleaned.append(text)

    return cleaned


def _normalize_for_comparison(line: str) -> str:
    """Normalize a line for header/footer comparison — lowercase, collapse spaces, remove digits."""
    line = line.lower().strip()
    line = re.sub(r"\d+", "#", line)  # Replace digits with # to match "Page 1" == "Page 2"
    line = re.sub(r"\s+", " ", line)
    return line
