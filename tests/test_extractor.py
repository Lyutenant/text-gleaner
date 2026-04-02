from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from textgleaner.extractor import extract, _check_size


class TestCheckSize:
    def test_within_limit_passes(self, tmp_path):
        _check_size("hello", tmp_path / "f.txt", 100)

    def test_exceeds_limit_raises(self, tmp_path):
        with pytest.raises(ValueError, match="max_chars"):
            _check_size("x" * 101, tmp_path / "f.txt", 100)

    def test_zero_limit_disabled(self, tmp_path):
        _check_size("x" * 1_000_000, tmp_path / "f.txt", 0)


class TestExtract:
    def _schema(self):
        return {
            "name": "extract_data",
            "description": "Test",
            "parameters": {"type": "object", "properties": {
                "account_number": {"type": ["string", "null"], "description": "Account #"},
            }},
        }

    def test_single_input_returns_flat_dict(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("Account: 12345")

        mock_client = MagicMock()
        mock_client.chat.return_value = {}
        mock_client.get_tool_arguments.return_value = {"account_number": "12345"}

        with patch("textgleaner.extractor.LLMClient", return_value=mock_client):
            result = extract([f], self._schema(), None, single=True)

        assert result == {"account_number": "12345"}

    def test_multiple_inputs_returns_keyed_dict(self, tmp_path):
        f1, f2 = tmp_path / "jan.txt", tmp_path / "feb.txt"
        f1.write_text("Account: 111")
        f2.write_text("Account: 222")

        mock_client = MagicMock()
        mock_client.chat.return_value = {}
        mock_client.get_tool_arguments.side_effect = [
            {"account_number": "111"},
            {"account_number": "222"},
        ]

        with patch("textgleaner.extractor.LLMClient", return_value=mock_client):
            result = extract([f1, f2], self._schema(), None, single=False)

        assert result == {
            "jan.txt": {"account_number": "111"},
            "feb.txt": {"account_number": "222"},
        }

    def test_output_written_for_single(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("Account: 12345")
        out = tmp_path / "result.json"

        mock_client = MagicMock()
        mock_client.chat.return_value = {}
        mock_client.get_tool_arguments.return_value = {"account_number": "12345"}

        with patch("textgleaner.extractor.LLMClient", return_value=mock_client):
            extract([f], self._schema(), out, single=True)

        assert out.exists()
        assert json.loads(out.read_text())["account_number"] == "12345"

    def test_size_limit_enforced(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("x" * 1000)

        with pytest.raises(ValueError, match="max_chars"):
            extract([f], self._schema(), None, single=True, max_chars=100)

    def test_size_limit_kwarg_overrides_default(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("x" * 300_000)

        mock_client = MagicMock()
        mock_client.chat.return_value = {}
        mock_client.get_tool_arguments.return_value = {"account_number": "x"}

        with patch("textgleaner.extractor.LLMClient", return_value=mock_client):
            # Default is 200_000 — would fail without override
            result = extract([f], self._schema(), None, single=True, max_chars=0)

        assert "account_number" in result


class TestPublicAPI:
    def _schema(self):
        return {
            "name": "extract_data",
            "description": "Test",
            "parameters": {"type": "object", "properties": {
                "value": {"type": ["string", "null"], "description": "Value"},
            }},
        }

    def test_single_path_string(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("Value: 42")

        mock_client = MagicMock()
        mock_client.chat.return_value = {}
        mock_client.get_tool_arguments.return_value = {"value": "42"}

        with patch("textgleaner.extractor.LLMClient", return_value=mock_client):
            from textgleaner import extract
            result = extract(str(f), schema=self._schema())

        assert result == {"value": "42"}

    def test_base_url_kwarg_passed_to_client(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("Value: 99")

        mock_client = MagicMock()
        mock_client.chat.return_value = {}
        mock_client.get_tool_arguments.return_value = {"value": "99"}

        with patch("textgleaner.extractor.LLMClient", return_value=mock_client) as MockClient:
            from textgleaner import extract
            extract(str(f), schema=self._schema(), base_url="http://custom:9999")

        _, kwargs = MockClient.call_args
        assert kwargs.get("base_url") == "http://custom:9999"
