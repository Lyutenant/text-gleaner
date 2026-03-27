from __future__ import annotations
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pdfetch.pdf_reader import extract_text_from_pdf, chunk_pages


class TestChunkPages:
    def test_no_chunking_when_size_zero(self):
        pages = ["page1", "page2", "page3"]
        result = chunk_pages(pages, 0, 0)
        assert len(result) == 1
        assert result[0][0] == 0
        assert result[0][1] == 3
        assert "page1" in result[0][2]

    def test_no_chunking_when_pages_within_limit(self):
        pages = ["a", "b", "c"]
        result = chunk_pages(pages, 10, 1)
        assert len(result) == 1

    def test_chunking_basic(self):
        pages = [f"page{i}" for i in range(10)]
        result = chunk_pages(pages, 4, 1)
        # Should produce multiple chunks
        assert len(result) > 1
        # First chunk starts at 0
        assert result[0][0] == 0
        # Last chunk ends at 10
        assert result[-1][1] == 10

    def test_chunking_overlap(self):
        pages = [f"page{i}" for i in range(9)]
        chunks = chunk_pages(pages, 3, 1)
        # step = 3 - 1 = 2; starts: 0, 2, 4, 6, 8
        starts = [c[0] for c in chunks]
        assert starts[0] == 0
        assert starts[1] == 2

    def test_empty_pages_excluded_from_text(self):
        pages = ["real content", "", "  ", "more content"]
        result = chunk_pages(pages, 0, 0)
        assert "" not in result[0][2].split("\n\n")
        assert "real content" in result[0][2]


class TestExtractTextFromPdf:
    def test_normal_extraction(self, tmp_path):
        """Test successful pypdf extraction."""
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Hello world"
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page, mock_page]

        with patch("pypdf.PdfReader", return_value=mock_reader):
            result = extract_text_from_pdf(tmp_path / "test.pdf")

        assert result == ["Hello world", "Hello world"]

    def test_empty_page_fallback_to_pymupdf(self, tmp_path):
        """Pages with no pypdf text should fall back to pymupdf."""
        mock_page_empty = MagicMock()
        mock_page_empty.extract_text.return_value = ""
        mock_page_full = MagicMock()
        mock_page_full.extract_text.return_value = "Good text"
        mock_reader = MagicMock()
        mock_reader.pages = [mock_page_full, mock_page_empty]

        mock_fitz_page = MagicMock()
        mock_fitz_page.get_text.return_value = "Scanned text"
        mock_doc = MagicMock()
        mock_doc.__getitem__ = lambda self, i: mock_fitz_page
        mock_doc.__len__ = lambda self: 2

        with patch("pypdf.PdfReader", return_value=mock_reader):
            with patch("fitz.open", return_value=mock_doc):
                result = extract_text_from_pdf(tmp_path / "test.pdf")

        assert result[0] == "Good text"
        assert result[1] == "Scanned text"

    def test_no_extractable_text_returns_empty_list(self, tmp_path):
        """When all extraction methods fail, return empty list."""
        with patch("pypdf.PdfReader", side_effect=Exception("corrupt")):
            with patch("fitz.open", side_effect=Exception("also failed")):
                result = extract_text_from_pdf(tmp_path / "bad.pdf")

        assert result == []
