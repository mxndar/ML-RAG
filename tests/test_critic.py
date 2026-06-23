"""Tests for the rule-based critic module."""

import pytest
from app.schemas import Chunk, ChunkMeta, Evidence, ExtractedTable
from app.retrieval.critic import (
    validate_evidence,
    _has_table_intent,
    _has_section_reference,
    _has_page_reference,
)


def _make_chunk(text="sample text", page=1, section="", score=0.5):
    return Chunk(
        text=text,
        metadata=ChunkMeta(
            chunk_id="test-chunk",
            page_number=page,
            section_name=section,
            document_name="test",
            source=f"test, page {page}",
        ),
        score=score,
    )


def _make_table(page=1):
    return ExtractedTable(
        page_number=page,
        headers=["Col1", "Col2"],
        rows=[["a", "b"]],
        markdown="| Col1 | Col2 |\n| --- | --- |\n| a | b |",
    )


class TestIntentDetection:
    def test_table_intent(self):
        assert _has_table_intent("What does table 3 show?")
        assert _has_table_intent("Show me the dosage data")
        assert _has_table_intent("Compare the results")
        assert not _has_table_intent("What is the conclusion?")

    def test_section_reference(self):
        assert _has_section_reference("What is in section 3.2?")
        assert _has_section_reference("Summarize chapter 5")
        assert _has_section_reference("See appendix A")
        assert not _has_section_reference("What is the main finding?")

    def test_page_reference(self):
        assert _has_page_reference("What is on page 42?")
        assert _has_page_reference("See pages 10")
        assert not _has_page_reference("What is the finding?")


class TestCritic:
    def test_complete_evidence(self):
        evidence = Evidence(
            chunks=[_make_chunk()],
            tables=[],
            total_tokens=50,
            confidence=0.7,
        )
        result = validate_evidence("What is the conclusion?", evidence)
        assert result.is_complete is True
        assert len(result.issues) == 0

    def test_no_evidence(self):
        evidence = Evidence(chunks=[], tables=[], total_tokens=0, confidence=0.0)
        result = validate_evidence("any query", evidence)
        assert result.is_complete is False
        assert "no_evidence_found" in result.issues

    def test_missing_table_data(self):
        evidence = Evidence(
            chunks=[_make_chunk("some text without structured data")],
            tables=[],
            total_tokens=50,
            confidence=0.5,
        )
        result = validate_evidence("What does table 3 show?", evidence)
        assert result.is_complete is False
        assert "missing_table_data" in result.issues

    def test_table_present_when_needed(self):
        evidence = Evidence(
            chunks=[_make_chunk()],
            tables=[_make_table()],
            total_tokens=50,
            confidence=0.5,
        )
        result = validate_evidence("What does table 3 show?", evidence)
        assert "missing_table_data" not in result.issues

    def test_missing_section_content(self):
        evidence = Evidence(
            chunks=[_make_chunk(section="")],
            tables=[],
            total_tokens=50,
            confidence=0.5,
        )
        result = validate_evidence("Summarize section 3.2", evidence)
        assert "missing_section_content" in result.issues

    def test_section_present(self):
        evidence = Evidence(
            chunks=[_make_chunk(section="3.2 Methods")],
            tables=[],
            total_tokens=50,
            confidence=0.5,
        )
        result = validate_evidence("Summarize section 3.2", evidence)
        assert "missing_section_content" not in result.issues

    def test_missing_page_content(self):
        evidence = Evidence(
            chunks=[_make_chunk(page=5)],
            tables=[],
            total_tokens=50,
            confidence=0.5,
        )
        result = validate_evidence("What is on page 42?", evidence)
        assert "missing_page_content" in result.issues

    def test_page_present(self):
        evidence = Evidence(
            chunks=[_make_chunk(page=42)],
            tables=[],
            total_tokens=50,
            confidence=0.5,
        )
        result = validate_evidence("What is on page 42?", evidence)
        assert "missing_page_content" not in result.issues
