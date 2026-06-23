"""Tests for the evidence merger module."""

import pytest
from app.schemas import Chunk, ChunkMeta, Evidence, ExtractedTable
from app.retrieval.evidence_merger import merge_evidence


def _make_chunk(text="sample text here", chunk_id="c1", page=1, section="", score=0.5):
    return Chunk(
        text=text,
        metadata=ChunkMeta(
            chunk_id=chunk_id,
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
        headers=["A", "B"],
        rows=[["1", "2"]],
        markdown="| A | B |\n| --- | --- |\n| 1 | 2 |",
    )


class TestEvidenceMerger:
    def test_empty_input(self):
        evidence = merge_evidence()
        assert len(evidence.chunks) == 0
        assert len(evidence.tables) == 0
        assert evidence.total_tokens == 0

    def test_bm25_only(self):
        chunks = [_make_chunk(chunk_id="c1", score=0.8), _make_chunk(chunk_id="c2", score=0.6)]
        evidence = merge_evidence(bm25_chunks=chunks)
        assert len(evidence.chunks) == 2
        assert evidence.total_tokens > 0

    def test_deduplication(self):
        chunk = _make_chunk(chunk_id="same-id")
        evidence = merge_evidence(bm25_chunks=[chunk], page_chunks=[chunk])
        assert len(evidence.chunks) == 1

    def test_token_limit(self):
        # Create chunks with enough text to exceed 300 tokens
        big_chunks = [
            _make_chunk(text=" ".join(["word"] * 200), chunk_id=f"c{i}")
            for i in range(5)
        ]
        evidence = merge_evidence(bm25_chunks=big_chunks, max_tokens=300)
        assert evidence.total_tokens <= 300

    def test_priority_order(self):
        bm25 = _make_chunk(chunk_id="bm25", text="bm25 result", score=0.9)
        page = _make_chunk(chunk_id="page", text="page result", score=0.0)
        evidence = merge_evidence(bm25_chunks=[bm25], page_chunks=[page], max_tokens=300)
        # BM25 should come first
        assert evidence.chunks[0].metadata.chunk_id == "bm25"

    def test_tables_included(self):
        chunks = [_make_chunk()]
        tables = [_make_table()]
        evidence = merge_evidence(bm25_chunks=chunks, tables=tables)
        assert len(evidence.tables) == 1

    def test_confidence_calculation(self):
        chunks = [_make_chunk(score=0.8)]
        evidence = merge_evidence(bm25_chunks=chunks)
        assert evidence.confidence > 0
        assert evidence.confidence <= 1.0
