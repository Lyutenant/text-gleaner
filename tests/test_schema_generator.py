from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from textgleaner.schema_generator import _parse_schema_json, _validate_schema, generate_schema


class TestParseSchemaJson:
    def test_plain_json(self):
        raw = '{"name": "test", "description": "d", "parameters": {"properties": {}}}'
        assert _parse_schema_json(raw)["name"] == "test"

    def test_strips_markdown_fences(self):
        raw = '```json\n{"name": "x", "description": "d", "parameters": {"properties": {}}}\n```'
        assert _parse_schema_json(raw)["name"] == "x"

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            _parse_schema_json("not json")


class TestValidateSchema:
    def test_valid_schema_passes(self):
        _validate_schema({
            "name": "extract_data",
            "description": "Extract data",
            "parameters": {"type": "object", "properties": {}},
        })

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
    def _schema(self):
        return {
            "name": "extract_statement",
            "description": "Statement",
            "parameters": {"type": "object", "properties": {
                "account_number": {"type": ["string", "null"], "description": "Account number"},
            }},
        }

    def test_generates_and_writes_schema(self, tmp_path):
        sample = tmp_path / "sample.txt"
        sample.write_text("Account: 12345\nValue: $1000")
        output = tmp_path / "schema.json"
        schema = self._schema()

        mock_client = MagicMock()
        mock_client.chat.return_value = {}
        mock_client.get_content.return_value = json.dumps(schema)

        with patch("textgleaner.schema_generator.LLMClient", return_value=mock_client):
            result = generate_schema([sample], "Test document", output)

        assert result["name"] == "extract_statement"
        assert output.exists()

    def test_base_url_kwarg_passed_to_client(self, tmp_path):
        sample = tmp_path / "sample.txt"
        sample.write_text("some text")
        schema = self._schema()

        mock_client = MagicMock()
        mock_client.chat.return_value = {}
        mock_client.get_content.return_value = json.dumps(schema)

        with patch("textgleaner.schema_generator.LLMClient", return_value=mock_client) as MockClient:
            generate_schema([sample], "Test", None, base_url="http://custom:9999")

        _, kwargs = MockClient.call_args
        assert kwargs.get("base_url") == "http://custom:9999"

    def test_retries_on_invalid_json(self, tmp_path):
        sample = tmp_path / "sample.txt"
        sample.write_text("some text")
        output = tmp_path / "schema.json"
        schema = self._schema()

        mock_client = MagicMock()
        mock_client.chat.return_value = {}
        mock_client.get_content.side_effect = ["not valid json", json.dumps(schema)]

        with patch("textgleaner.schema_generator.LLMClient", return_value=mock_client):
            result = generate_schema([sample], "Test", output)

        assert result["name"] == "extract_statement"
        assert mock_client.chat.call_count == 2

    def test_empty_sample_raises(self, tmp_path):
        sample = tmp_path / "empty.txt"
        sample.write_text("")

        with pytest.raises(ValueError, match="No readable text"):
            generate_schema([sample], "Test", None)
