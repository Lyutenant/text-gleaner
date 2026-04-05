from __future__ import annotations
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import httpx
from textgleaner.extractor import (
    extract, _check_size, _extract_one_tool_call, _extract_one_structured,
    _build_retry_schema, _retry_low_confidence, RETRY_CONFIDENCE_THRESHOLD,
)


class TestCheckSize:
    def test_within_limit_passes(self):
        _check_size("hello", "f.txt", 100)

    def test_exceeds_limit_raises(self):
        with pytest.raises(ValueError, match="max_chars"):
            _check_size("x" * 101, "f.txt", 100)

    def test_zero_limit_disabled(self):
        _check_size("x" * 1_000_000, "f.txt", 0)


class TestExtract:
    def _schema(self):
        return {
            "name": "extract_data",
            "description": "Test",
            "parameters": {"type": "object", "properties": {
                "account_number": {"type": ["string", "null"], "description": "Account #"},
            }},
        }

    def test_single_input_returns_flat_dict(self):
        mock_client = MagicMock()
        mock_client.chat.return_value = {}
        mock_client.get_tool_arguments.return_value = {"account_number": "12345"}

        with patch("textgleaner.extractor.LLMClient", return_value=mock_client):
            result = extract([("Account: 12345", "doc.txt")], self._schema(), None, single=True)

        assert result == {"account_number": "12345"}

    def test_multiple_inputs_returns_keyed_dict(self):
        mock_client = MagicMock()
        mock_client.chat.return_value = {}
        mock_client.get_tool_arguments.side_effect = [
            {"account_number": "111"},
            {"account_number": "222"},
        ]

        with patch("textgleaner.extractor.LLMClient", return_value=mock_client):
            result = extract(
                [("Account: 111", "jan.txt"), ("Account: 222", "feb.txt")],
                self._schema(), None, single=False,
            )

        assert result == {
            "jan.txt": {"account_number": "111"},
            "feb.txt": {"account_number": "222"},
        }

    def test_output_written_for_single(self, tmp_path):
        out = tmp_path / "result.json"

        mock_client = MagicMock()
        mock_client.chat.return_value = {}
        mock_client.get_tool_arguments.return_value = {"account_number": "12345"}

        with patch("textgleaner.extractor.LLMClient", return_value=mock_client):
            extract([("Account: 12345", "doc.txt")], self._schema(), out, single=True)

        assert out.exists()
        assert json.loads(out.read_text())["account_number"] == "12345"

    def test_size_limit_enforced(self):
        with pytest.raises(ValueError, match="max_chars"):
            extract([("x" * 1000, "big.txt")], self._schema(), None, single=True, max_chars=100)

    def test_size_limit_kwarg_overrides_default(self):
        mock_client = MagicMock()
        mock_client.chat.return_value = {}
        mock_client.get_tool_arguments.return_value = {"account_number": "x"}

        with patch("textgleaner.extractor.LLMClient", return_value=mock_client):
            # Default is 200_000 — would fail without override
            result = extract([("x" * 300_000, "doc.txt")], self._schema(), None, single=True, max_chars=0)

        assert "account_number" in result


class TestExtractionMethods:
    def _schema(self):
        return {
            "name": "extract_data",
            "description": "Test",
            "parameters": {"type": "object", "properties": {
                "value": {"type": ["string", "null"], "description": "Value"},
            }},
        }

    def test_tool_call_method(self):
        mock_client = MagicMock()
        mock_client.chat.return_value = {}
        mock_client.get_tool_arguments.return_value = {"value": "42"}

        with patch("textgleaner.extractor.LLMClient", return_value=mock_client):
            result = extract([("text", "doc.txt")], self._schema(), None, single=True,
                             extraction_method="tool_call")

        assert result == {"value": "42"}
        # tool_call path calls get_tool_arguments, not get_content
        mock_client.get_tool_arguments.assert_called_once()
        mock_client.get_content.assert_not_called()

    def test_structured_output_method(self):
        mock_client = MagicMock()
        mock_client.chat.return_value = {}
        mock_client.get_content.return_value = '{"value": "99"}'

        with patch("textgleaner.extractor.LLMClient", return_value=mock_client):
            result = extract([("text", "doc.txt")], self._schema(), None, single=True,
                             extraction_method="structured_output")

        assert result == {"value": "99"}
        # structured_output path calls get_content, not get_tool_arguments
        mock_client.get_content.assert_called_once()
        mock_client.get_tool_arguments.assert_not_called()

    def test_structured_output_strips_markdown_fences(self):
        mock_client = MagicMock()
        mock_client.chat.return_value = {}
        mock_client.get_content.return_value = '```json\n{"value": "42"}\n```'

        with patch("textgleaner.extractor.LLMClient", return_value=mock_client):
            result = extract([("text", "doc.txt")], self._schema(), None, single=True,
                             extraction_method="structured_output")

        assert result == {"value": "42"}

    def test_structured_output_passes_response_format(self):
        mock_client = MagicMock()
        mock_client.chat.return_value = {}
        mock_client.get_content.return_value = '{"value": "x"}'

        with patch("textgleaner.extractor.LLMClient", return_value=mock_client):
            extract([("text", "doc.txt")], self._schema(), None, single=True,
                    extraction_method="structured_output")

        _, kwargs = mock_client.chat.call_args
        assert "response_format" in kwargs
        assert kwargs["response_format"]["type"] == "json_schema"
        assert "tools" not in kwargs or kwargs.get("tools") is None

    def test_auto_uses_tool_call_path(self):
        mock_client = MagicMock()
        mock_client.chat.return_value = {}
        mock_client.get_tool_arguments.return_value = {"value": "auto"}

        with patch("textgleaner.extractor.LLMClient", return_value=mock_client):
            result = extract([("text", "doc.txt")], self._schema(), None, single=True,
                             extraction_method="auto")

        assert result == {"value": "auto"}
        mock_client.get_tool_arguments.assert_called_once()

    def test_auto_falls_back_on_value_error(self):
        mock_client = MagicMock()
        mock_client.chat.return_value = {}
        mock_client.get_tool_arguments.side_effect = ValueError("No tool_calls and no content")
        mock_client.get_content.return_value = '{"value": "fallback"}'

        with patch("textgleaner.extractor.LLMClient", return_value=mock_client):
            result = extract([("text", "doc.txt")], self._schema(), None, single=True,
                             extraction_method="auto")

        assert result == {"value": "fallback"}
        mock_client.get_tool_arguments.assert_called_once()  # tried tool_call
        mock_client.get_content.assert_called_once()         # fell back to structured_output

    def test_auto_falls_back_on_http_400(self):
        mock_response = MagicMock()
        mock_response.status_code = 400
        http_error = httpx.HTTPStatusError("Bad Request", request=MagicMock(), response=mock_response)

        mock_client = MagicMock()
        # First chat call (tool_call) raises 400; second (structured_output) succeeds
        mock_client.chat.side_effect = [http_error, {}]
        mock_client.get_content.return_value = '{"value": "fallback"}'

        with patch("textgleaner.extractor.LLMClient", return_value=mock_client):
            result = extract([("text", "doc.txt")], self._schema(), None, single=True,
                             extraction_method="auto")

        assert result == {"value": "fallback"}
        assert mock_client.chat.call_count == 2
        mock_client.get_content.assert_called_once()

    def test_auto_reraises_non_4xx_http_errors(self):
        mock_response = MagicMock()
        mock_response.status_code = 500
        http_error = httpx.HTTPStatusError("Server Error", request=MagicMock(), response=mock_response)

        mock_client = MagicMock()
        mock_client.chat.side_effect = http_error

        with patch("textgleaner.extractor.LLMClient", return_value=mock_client):
            with pytest.raises(httpx.HTTPStatusError):
                extract([("text", "doc.txt")], self._schema(), None, single=True,
                        extraction_method="auto")

        mock_client.get_content.assert_not_called()  # no fallback attempted


class TestRetryLowConfidence:
    def _schema(self):
        return {
            "name": "extract_data",
            "description": "Test",
            "parameters": {"type": "object", "properties": {
                "account": {"type": ["string", "null"], "description": "Account"},
                "account_confidence": {"type": "number"},
                "amount": {"type": ["string", "null"], "description": "Amount"},
                "amount_confidence": {"type": "number"},
            }},
        }

    # --- _build_retry_schema ---

    def test_build_retry_schema_keeps_requested_fields(self):
        schema = self._schema()
        retry = _build_retry_schema(schema, ["amount"])
        props = retry["parameters"]["properties"]
        assert "amount" in props
        assert "amount_confidence" in props

    def test_build_retry_schema_excludes_other_fields(self):
        schema = self._schema()
        retry = _build_retry_schema(schema, ["amount"])
        props = retry["parameters"]["properties"]
        assert "account" not in props
        assert "account_confidence" not in props

    def test_build_retry_schema_preserves_name(self):
        schema = self._schema()
        retry = _build_retry_schema(schema, ["amount"])
        assert retry["name"] == schema["name"]

    def test_build_retry_schema_no_confidence_sibling(self):
        schema = {
            "name": "x",
            "description": "",
            "parameters": {"type": "object", "properties": {
                "val": {"type": ["string", "null"]},
            }},
        }
        retry = _build_retry_schema(schema, ["val"])
        assert "val" in retry["parameters"]["properties"]
        assert "val_confidence" not in retry["parameters"]["properties"]

    # --- _retry_low_confidence ---

    def test_no_retry_when_all_confident(self):
        """No LLM call when every field is above the threshold."""
        mock_client = MagicMock()
        result = {"account": "123", "account_confidence": 1.0,
                  "amount": "50", "amount_confidence": 0.9}
        updated = _retry_low_confidence(mock_client, self._schema(), "text", "doc.txt", result, "tool_call")
        mock_client.chat.assert_not_called()
        assert updated == result

    def test_no_retry_when_no_confidence_fields(self):
        """Fields without _confidence siblings are skipped (confidence_scores disabled)."""
        mock_client = MagicMock()
        result = {"account": None, "amount": None}
        updated = _retry_low_confidence(mock_client, self._schema(), "text", "doc.txt", result, "tool_call")
        mock_client.chat.assert_not_called()
        assert updated == result

    def test_retry_called_for_low_confidence_field(self):
        mock_client = MagicMock()
        mock_client.chat.return_value = {}
        mock_client.get_tool_arguments.return_value = {
            "amount": "99.00", "amount_confidence": 0.9,
        }
        result = {"account": "123", "account_confidence": 1.0,
                  "amount": None, "amount_confidence": 0.0}
        updated = _retry_low_confidence(mock_client, self._schema(), "text", "doc.txt", result, "tool_call")
        mock_client.chat.assert_called_once()
        assert updated["amount"] == "99.00"
        assert updated["amount_confidence"] == 0.9

    def test_retry_at_threshold_boundary(self):
        """A field exactly at the threshold (0.4) is retried."""
        mock_client = MagicMock()
        mock_client.chat.return_value = {}
        mock_client.get_tool_arguments.return_value = {
            "amount": "50.00", "amount_confidence": 0.7,
        }
        result = {"amount": "~50", "amount_confidence": RETRY_CONFIDENCE_THRESHOLD,
                  "amount_confidence": 0.4}
        updated = _retry_low_confidence(mock_client, self._schema(), "text", "doc.txt", result, "tool_call")
        mock_client.chat.assert_called_once()

    def test_retry_does_not_downgrade(self):
        """If the retry returns equal or lower confidence, the original value is kept."""
        mock_client = MagicMock()
        mock_client.chat.return_value = {}
        mock_client.get_tool_arguments.return_value = {
            "amount": "wrong", "amount_confidence": 0.0,
        }
        result = {"account": "123", "account_confidence": 1.0,
                  "amount": "50", "amount_confidence": 0.4}
        updated = _retry_low_confidence(mock_client, self._schema(), "text", "doc.txt", result, "tool_call")
        # retry confidence (0.0) ≤ original (0.4) → keep original
        assert updated["amount"] == "50"
        assert updated["amount_confidence"] == 0.4

    def test_retry_failure_keeps_original(self):
        """If the retry call raises, the original result is returned unchanged."""
        mock_client = MagicMock()
        mock_client.chat.side_effect = ValueError("LLM error")
        result = {"amount": None, "amount_confidence": 0.0}
        updated = _retry_low_confidence(mock_client, self._schema(), "text", "doc.txt", result, "tool_call")
        assert updated == result

    def test_retry_only_sends_narrowed_schema(self):
        """The retry call must only include the low-confidence field, not the full schema."""
        mock_client = MagicMock()
        mock_client.chat.return_value = {}
        mock_client.get_tool_arguments.return_value = {"amount": "99", "amount_confidence": 0.8}
        result = {"account": "123", "account_confidence": 1.0,
                  "amount": None, "amount_confidence": 0.0}
        _retry_low_confidence(mock_client, self._schema(), "text", "doc.txt", result, "tool_call")
        # Inspect the tools payload sent to chat
        call_kwargs = mock_client.chat.call_args
        tools_sent = call_kwargs[1].get("tools") or call_kwargs[0][1]
        # The tool should only contain amount and amount_confidence properties
        props = tools_sent[0]["function"]["parameters"]["properties"]
        assert "amount" in props
        assert "account" not in props

    # --- extract() integration ---

    def test_extract_calls_retry_when_enabled(self):
        mock_client = MagicMock()
        mock_client.chat.return_value = {}
        # First call: initial extraction returns low-confidence amount
        # Second call (retry): returns improved amount
        mock_client.get_tool_arguments.side_effect = [
            {"amount": None, "amount_confidence": 0.0},          # initial
            {"amount": "42.00", "amount_confidence": 0.9},       # retry
        ]
        schema = {
            "name": "extract_data", "description": "",
            "parameters": {"type": "object", "properties": {
                "amount": {"type": ["string", "null"]},
                "amount_confidence": {"type": "number"},
            }},
        }
        with patch("textgleaner.extractor.LLMClient", return_value=mock_client):
            result = extract([("text", "doc.txt")], schema, None, single=True,
                             confidence_retry=True)
        assert result["amount"] == "42.00"
        assert mock_client.chat.call_count == 2

    def test_extract_no_retry_by_default(self):
        mock_client = MagicMock()
        mock_client.chat.return_value = {}
        mock_client.get_tool_arguments.return_value = {"amount": None, "amount_confidence": 0.0}
        schema = {
            "name": "extract_data", "description": "",
            "parameters": {"type": "object", "properties": {
                "amount": {"type": ["string", "null"]},
                "amount_confidence": {"type": "number"},
            }},
        }
        with patch("textgleaner.extractor.LLMClient", return_value=mock_client):
            extract([("text", "doc.txt")], schema, None, single=True)
        # Only one chat call — no retry
        assert mock_client.chat.call_count == 1


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

    def test_text_instance_single(self):
        mock_client = MagicMock()
        mock_client.chat.return_value = {}
        mock_client.get_tool_arguments.return_value = {"value": "99"}

        with patch("textgleaner.extractor.LLMClient", return_value=mock_client):
            from textgleaner import extract, Text
            result = extract(Text("Value: 99", name="section"), schema=self._schema())

        assert result == {"value": "99"}

    def test_text_instance_uses_name_as_key(self):
        mock_client = MagicMock()
        mock_client.chat.return_value = {}
        mock_client.get_tool_arguments.side_effect = [{"value": "a"}, {"value": "b"}]

        with patch("textgleaner.extractor.LLMClient", return_value=mock_client):
            from textgleaner import extract, Text
            result = extract(
                [Text("...", name="holdings"), Text("...", name="activities")],
                schema=self._schema(),
            )

        assert set(result.keys()) == {"holdings", "activities"}

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
