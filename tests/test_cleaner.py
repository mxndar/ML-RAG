"""Tests for the text cleaner module."""

import pytest
from app.ingestion.cleaner import (
    clean_text,
    remove_page_numbers,
    detect_headers_footers,
    remove_headers_footers,
    clean_document,
)


class TestCleanText:
    def test_removes_duplicate_spaces(self):
        assert clean_text("hello   world") == "hello world"

    def test_collapses_multiple_newlines(self):
        result = clean_text("line1\n\n\n\nline2")
        assert result == "line1\n\nline2"

    def test_strips_whitespace(self):
        assert clean_text("  hello  ") == "hello"

    def test_preserves_single_newlines(self):
        result = clean_text("line1\nline2")
        assert "line1\nline2" == result

    def test_handles_empty_string(self):
        assert clean_text("") == ""

    def test_unicode_normalization(self):
        # NFKC normalizes ﬁ → fi
        result = clean_text("ﬁnd")
        assert "fi" in result

    def test_preserves_stop_words(self):
        text = "this is a test of the system and it works for us"
        result = clean_text(text)
        # All stop words should be preserved
        for word in ["this", "is", "a", "of", "the", "and", "it", "for", "us"]:
            assert word in result


class TestRemovePageNumbers:
    def test_removes_standalone_number(self):
        result = remove_page_numbers("  42  ")
        assert result.strip() == ""

    def test_removes_page_x_format(self):
        result = remove_page_numbers("Page 42")
        assert result.strip() == ""

    def test_removes_page_x_of_y(self):
        result = remove_page_numbers("page 5 of 100")
        assert result.strip() == ""

    def test_removes_dash_format(self):
        result = remove_page_numbers("- 42 -")
        assert result.strip() == ""

    def test_preserves_numbers_in_text(self):
        result = remove_page_numbers("There are 42 items in the list")
        assert "42" in result


class TestDetectHeadersFooters:
    def test_detects_repeated_header(self):
        pages = [f"Company Report 2024\nContent page {i}" for i in range(10)]
        headers, footers = detect_headers_footers(pages)
        # "company report #" should be detected
        assert len(headers) > 0

    def test_no_detection_few_pages(self):
        pages = ["Content 1", "Content 2"]
        headers, footers = detect_headers_footers(pages)
        assert len(headers) == 0
        assert len(footers) == 0


class TestCleanDocument:
    def test_cleans_multi_page_doc(self):
        pages = [
            "Header Line\nThis is unique content A.\n1",
            "Header Line\nThis is unique content B.\n2",
            "Header Line\nThis is unique content C.\n3",
            "Header Line\nThis is unique content D.\n4",
            "Header Line\nThis is unique content E.\n5",
            "Header Line\nThis is unique content F.\n6",
        ]
        cleaned = clean_document(pages)
        assert len(cleaned) == 6
        # Content should be preserved
        for page in cleaned:
            assert "content" in page.lower()
