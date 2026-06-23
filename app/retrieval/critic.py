"""
Critic — rule-based evidence validator. No LLM.
Checks if the evidence is sufficient for the query.
Triggers planner retry if incomplete.
"""

from __future__ import annotations

import logging
import re

from app.schemas import CriticResult, Evidence

logger = logging.getLogger(__name__)


# ── Intent detection helpers ────────────────────────────────

def _has_table_intent(query: str) -> bool:
    """Check if the query asks about tabular/numerical data."""
    table_keywords = [
        r"\btable\b", r"\btables\b", r"\bfigure\b",
        r"\bdata\b", r"\bstatistics\b", r"\bstat\b",
        r"\bdosage\b", r"\bdose\b", r"\bnumbers?\b",
        r"\bvalues?\b", r"\bpercentage\b", r"\bcompare\b",
        r"\bcomparison\b", r"\bresults?\b", r"\bmetrics?\b",
        r"\blist\b", r"\bcolumn\b", r"\brow\b",
    ]
    pattern = "|".join(table_keywords)
    return bool(re.search(pattern, query.lower()))


def _has_section_reference(query: str) -> bool:
    """Check if the query references a specific section or chapter."""
    section_patterns = [
        r"\bsection\s+[\d.]+",
        r"\bchapter\s+\d+",
        r"\bpart\s+\d+",
        r"\bappendix\s+[a-z\d]",
        r"\bsection\s+['\"]",
        r"\bin\s+the\s+\w+\s+section\b",
    ]
    pattern = "|".join(section_patterns)
    return bool(re.search(pattern, query.lower()))


def _has_page_reference(query: str) -> bool:
    """Check if the query mentions specific pages."""
    page_patterns = [
        r"\bpage\s+\d+",
        r"\bpages?\s+\d+",
        r"\bp\.\s*\d+",
        r"\bon\s+page\b",
    ]
    pattern = "|".join(page_patterns)
    return bool(re.search(pattern, query.lower()))


def _has_section_evidence(evidence: Evidence) -> bool:
    """Check if any evidence chunk comes from a named section."""
    return any(
        chunk.metadata.section_name
        for chunk in evidence.chunks
    )


def _has_page_evidence(evidence: Evidence, query: str) -> bool:
    """Check if evidence includes content from the referenced page."""
    # Extract page numbers from query
    page_numbers = [int(m) for m in re.findall(r"page\s+(\d+)", query.lower())]
    if not page_numbers:
        return True  # No specific page referenced

    evidence_pages = {chunk.metadata.page_number for chunk in evidence.chunks}
    return any(p in evidence_pages for p in page_numbers)


# ── Main critic function ───────────────────────────────────

def validate_evidence(query: str, evidence: Evidence) -> CriticResult:
    """
    Rule-based validation of retrieved evidence.

    Checks:
      1. Is there any evidence at all?
      2. Is confidence too low?
      3. Does the query mention tables but no tables were retrieved?
      4. Does the query reference a section but no section content was found?
      5. Does the query mention a page but that page isn't in the evidence?

    Returns:
        CriticResult with is_complete=True if evidence is sufficient,
        or is_complete=False with a list of issues.
    """
    issues: list[str] = []

    # Check 1: Any evidence at all?
    if not evidence.chunks and not evidence.tables:
        issues.append("no_evidence_found")
        return CriticResult(is_complete=False, issues=issues)

    # Check 2: Confidence too low?
    if evidence.confidence < 0.15:
        issues.append("low_confidence")

    # Check 3: Table intent but no tables?
    if _has_table_intent(query) and not evidence.tables:
        # Only flag if no chunks mention table-like content either
        chunk_text = " ".join(c.text for c in evidence.chunks).lower()
        if "table" not in chunk_text and "|" not in chunk_text:
            issues.append("missing_table_data")

    # Check 4: Section reference but no section content?
    if _has_section_reference(query) and not _has_section_evidence(evidence):
        issues.append("missing_section_content")

    # Check 5: Page reference but wrong page?
    if _has_page_reference(query) and not _has_page_evidence(evidence, query):
        issues.append("missing_page_content")

    is_complete = len(issues) == 0

    if not is_complete:
        logger.info(f"Critic: INCOMPLETE — issues={issues}")
    else:
        logger.info("Critic: COMPLETE ✓")

    return CriticResult(is_complete=is_complete, issues=issues)
