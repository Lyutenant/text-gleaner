from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from pdfetch.utils import merge_chunks, null_rate


class TestMergeChunks:
    def test_empty_returns_empty(self):
        assert merge_chunks([]) == {}

    def test_single_chunk_returned_as_is(self):
        chunk = {"name": "Alice", "amount": 100}
        assert merge_chunks([chunk]) == chunk

    def test_later_chunk_fills_nulls(self):
        chunk1 = {"name": "Alice", "amount": None}
        chunk2 = {"name": None, "amount": 200}
        result = merge_chunks([chunk1, chunk2])
        assert result["name"] == "Alice"   # chunk1 had value
        assert result["amount"] == 200     # chunk1 had null, chunk2 fills it

    def test_earlier_value_not_overwritten(self):
        chunk1 = {"name": "Alice"}
        chunk2 = {"name": "Bob"}
        result = merge_chunks([chunk1, chunk2])
        assert result["name"] == "Alice"

    def test_three_chunks(self):
        c1 = {"a": "first", "b": None, "c": None}
        c2 = {"a": None, "b": "second", "c": None}
        c3 = {"a": None, "b": None, "c": "third"}
        result = merge_chunks([c1, c2, c3])
        assert result == {"a": "first", "b": "second", "c": "third"}


class TestNullRate:
    def test_empty_records(self):
        assert null_rate([]) == {}

    def test_all_null(self):
        records = [{"field": None}, {"field": None}]
        rates = null_rate(records)
        assert rates["field"] == 1.0

    def test_no_nulls(self):
        records = [{"field": "a"}, {"field": "b"}]
        rates = null_rate(records)
        assert rates["field"] == 0.0

    def test_partial_nulls(self):
        records = [{"field": "a"}, {"field": None}, {"field": "c"}, {"field": None}]
        rates = null_rate(records)
        assert rates["field"] == 0.5

    def test_confidence_fields_excluded(self):
        records = [{"amount": None, "amount_confidence": 0.0}]
        rates = null_rate(records)
        assert "amount_confidence" not in rates
        assert "amount" in rates


class TestExtractIntegration:
    """Integration-style tests for the extractor using mocked LLM."""

    def _make_schema(self):
        return {
            "name": "extract_data",
            "description": "Test extraction",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_number": {"type": ["string", "null"], "description": "Account #"},
                },
            },
        }

    def test_extract_single_pdf(self, tmp_path):
        from pdfetch.extractor import extract

        pdfs_dir = tmp_path / "pdfs"
        pdfs_dir.mkdir()
        fake_pdf = pdfs_dir / "statement.pdf"
        fake_pdf.write_bytes(b"%PDF fake")

        schema_file = tmp_path / "schema.json"
        schema = self._make_schema()
        schema_file.write_text(json.dumps(schema))

        output_dir = tmp_path / "output"
        extracted_data = {"account_number": "12345678"}

        mock_client = MagicMock()
        mock_client.chat.return_value = {}
        mock_client.get_tool_arguments.return_value = extracted_data

        with patch("pdfetch.extractor.LLMClient", return_value=mock_client):
            with patch("pdfetch.extractor.extract_text_from_pdf", return_value=["page text"]):
                # Patch config to disable dry_run_first
                from pdfetch.config import AppConfig, LLMConfig, ExtractionConfig
                mock_cfg = MagicMock()
                mock_cfg.extraction.dry_run_first = False
                mock_cfg.extraction.chunk_size_pages = 0
                mock_cfg.extraction.chunk_overlap_pages = 0
                mock_cfg.extraction.output_mode = "per_file"

                with patch("pdfetch.extractor.get_config", return_value=mock_cfg):
                    extract(pdfs_dir, schema_file, output_dir, dry_run=False)

        out_file = output_dir / "statement.json"
        assert out_file.exists()
        result = json.loads(out_file.read_text())
        assert result["account_number"] == "12345678"
