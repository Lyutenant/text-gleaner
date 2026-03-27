from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pdfetch.schema_generator import _parse_schema_json, _validate_schema, generate_schema


class TestParseSchemaJson:
    def test_plain_json(self):
        raw = '{"name": "test", "description": "d", "parameters": {"properties": {}}}'
        result = _parse_schema_json(raw)
        assert result["name"] == "test"

    def test_strips_markdown_fences(self):
        raw = '```json\n{"name": "x", "description": "d", "parameters": {"properties": {}}}\n```'
        result = _parse_schema_json(raw)
        assert result["name"] == "x"

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_schema_json("not json")


class TestValidateSchema:
    def test_valid_schema_passes(self):
        schema = {
            "name": "extract_data",
            "description": "Extract data",
            "parameters": {"type": "object", "properties": {}},
        }
        _validate_schema(schema)  # Should not raise

    def test_missing_name_raises(self):
        with pytest.raises(ValueError, match="name"):
            _validate_schema({"description": "d", "parameters": {"properties": {}}})

    def test_missing_parameters_raises(self):
        with pytest.raises(ValueError, match="parameters"):
            _validate_schema({"name": "x", "description": "d"})

    def test_missing_properties_raises(self):
        with pytest.raises(ValueError, match="properties"):
            _validate_schema({"name": "x", "description": "d", "parameters": {}})


class TestGenerateSchema:
    def _make_valid_schema(self):
        return {
            "name": "extract_statement",
            "description": "Monthly brokerage statement",
            "parameters": {
                "type": "object",
                "properties": {
                    "account_number": {"type": ["string", "null"], "description": "Account number"},
                    "account_number_confidence": {"type": "number", "description": "Confidence"},
                },
            },
        }

    def test_generates_and_writes_schema(self, tmp_path):
        pdfs_dir = tmp_path / "pdfs"
        pdfs_dir.mkdir()
        # Create a fake PDF file
        fake_pdf = pdfs_dir / "sample.pdf"
        fake_pdf.write_bytes(b"%PDF-1.4 fake")

        desc_file = tmp_path / "desc.yaml"
        desc_file.write_text("document_type: Test\nkey_fields:\n  - account_number\n")

        output_file = tmp_path / "schema.json"
        schema = self._make_valid_schema()

        mock_client = MagicMock()
        mock_response = {"choices": [{"message": {"content": json.dumps(schema), "tool_calls": None}}]}
        mock_client.chat.return_value = mock_response
        mock_client.get_content.return_value = json.dumps(schema)

        with patch("pdfetch.schema_generator.LLMClient", return_value=mock_client):
            with patch("pdfetch.schema_generator._extract_snippet", return_value="sample text"):
                result = generate_schema(pdfs_dir, desc_file, output_file)

        assert result["name"] == "extract_statement"
        assert output_file.exists()
        saved = json.loads(output_file.read_text())
        assert saved["name"] == "extract_statement"

    def test_retries_on_invalid_json(self, tmp_path):
        pdfs_dir = tmp_path / "pdfs"
        pdfs_dir.mkdir()
        (pdfs_dir / "sample.pdf").write_bytes(b"%PDF fake")

        desc_file = tmp_path / "desc.yaml"
        desc_file.write_text("document_type: Test\n")

        output_file = tmp_path / "schema.json"
        schema = self._make_valid_schema()

        mock_client = MagicMock()
        # First call returns bad JSON, second returns valid
        mock_client.get_content.side_effect = ["not valid json", json.dumps(schema)]
        mock_client.chat.return_value = {}

        with patch("pdfetch.schema_generator.LLMClient", return_value=mock_client):
            with patch("pdfetch.schema_generator._extract_snippet", return_value="text"):
                result = generate_schema(pdfs_dir, desc_file, output_file)

        assert result["name"] == "extract_statement"
        assert mock_client.chat.call_count == 2
