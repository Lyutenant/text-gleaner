from __future__ import annotations
import json
from unittest.mock import MagicMock, patch

import pytest

from textgleaner.schema_refiner import (
    refine_schema,
    _detect_confidence_scores,
    _run_gap_analysis,
    _run_schema_refinement,
)


def _base_schema(with_confidence=False):
    props = {
        "invoice_number": {"type": ["string", "null"], "description": "Invoice number"},
        "amount": {"type": ["string", "null"], "description": "Total amount"},
    }
    if with_confidence:
        props["invoice_number_confidence"] = {
            "type": "number",
            "description": "Confidence 0-1",
        }
        props["amount_confidence"] = {
            "type": "number",
            "description": "Confidence 0-1",
        }
    return {
        "name": "extract_invoice",
        "description": "Extract invoice data",
        "parameters": {"type": "object", "properties": props},
    }


def _updated_schema():
    """A schema that adds a 'vendor' field to the base schema."""
    return {
        "name": "extract_invoice",
        "description": "Extract invoice data",
        "parameters": {"type": "object", "properties": {
            "invoice_number": {"type": ["string", "null"], "description": "Invoice number"},
            "amount": {"type": ["string", "null"], "description": "Total amount"},
            "vendor": {"type": ["string", "null"], "description": "Vendor name"},
        }},
    }


def _mock_client(gap_analysis="gap analysis text", updated_schema=None):
    if updated_schema is None:
        updated_schema = _updated_schema()
    mock = MagicMock()
    mock.chat.return_value = {}
    mock.get_content.side_effect = [
        gap_analysis,
        json.dumps(updated_schema),
    ]
    return mock


class TestDetectConfidenceScores:
    def test_detects_confidence_fields(self):
        assert _detect_confidence_scores(_base_schema(with_confidence=True)) is True

    def test_no_confidence_fields(self):
        assert _detect_confidence_scores(_base_schema(with_confidence=False)) is False

    def test_empty_schema(self):
        schema = {"name": "x", "description": "", "parameters": {"type": "object", "properties": {}}}
        assert _detect_confidence_scores(schema) is False


class TestRunGapAnalysis:
    def test_sends_schema_in_prompt(self):
        mock = MagicMock()
        mock.chat.return_value = {}
        mock.get_content.return_value = "analysis"

        _run_gap_analysis(mock, _base_schema(), "sample text")

        messages = mock.chat.call_args[0][0]
        user_msg = messages[1]["content"]
        assert "invoice_number" in user_msg   # schema is embedded
        assert "sample text" in user_msg

    def test_returns_analysis_string(self):
        mock = MagicMock()
        mock.chat.return_value = {}
        mock.get_content.return_value = "field X is missing"

        result = _run_gap_analysis(mock, _base_schema(), "text")
        assert result == "field X is missing"


class TestRunSchemaRefinement:
    def test_returns_valid_schema(self):
        mock = MagicMock()
        mock.chat.return_value = {}
        mock.get_content.return_value = json.dumps(_updated_schema())

        result = _run_schema_refinement(mock, _base_schema(), "analysis", False)
        assert result["name"] == "extract_invoice"
        assert "vendor" in result["parameters"]["properties"]

    def test_retry_on_bad_json(self):
        mock = MagicMock()
        mock.chat.return_value = {}
        mock.get_content.side_effect = [
            "not json at all",            # first attempt fails
            json.dumps(_updated_schema()),  # retry succeeds
        ]

        result = _run_schema_refinement(mock, _base_schema(), "analysis", False)
        assert mock.chat.call_count == 2
        assert "vendor" in result["parameters"]["properties"]

    def test_raises_after_two_failures(self):
        mock = MagicMock()
        mock.chat.return_value = {}
        mock.get_content.side_effect = ["not json", "also not json"]

        with pytest.raises(ValueError, match="failed after retry"):
            _run_schema_refinement(mock, _base_schema(), "analysis", False)

    def test_confidence_instruction_included_when_enabled(self):
        mock = MagicMock()
        mock.chat.return_value = {}
        mock.get_content.return_value = json.dumps(_updated_schema())

        _run_schema_refinement(mock, _base_schema(), "analysis", confidence_scores=True)

        system_msg = mock.chat.call_args[0][0][0]["content"]
        assert "confidence" in system_msg.lower()

    def test_confidence_instruction_excluded_when_disabled(self):
        mock = MagicMock()
        mock.chat.return_value = {}
        mock.get_content.return_value = json.dumps(_updated_schema())

        _run_schema_refinement(mock, _base_schema(), "analysis", confidence_scores=False)

        system_msg = mock.chat.call_args[0][0][0]["content"]
        # The confidence instruction placeholder should be empty
        assert "foo_confidence" not in system_msg


class TestRefineSchema:
    def test_two_llm_calls_made(self):
        mock = _mock_client()
        with patch("textgleaner.schema_refiner.LLMClient", return_value=mock):
            refine_schema(_base_schema(), [("sample text", "doc.txt")], None)
        assert mock.chat.call_count == 2

    def test_returns_updated_schema(self):
        mock = _mock_client()
        with patch("textgleaner.schema_refiner.LLMClient", return_value=mock):
            result = refine_schema(_base_schema(), [("sample text", "doc.txt")], None)
        assert "vendor" in result["parameters"]["properties"]

    def test_existing_fields_preserved(self):
        mock = _mock_client()
        with patch("textgleaner.schema_refiner.LLMClient", return_value=mock):
            result = refine_schema(_base_schema(), [("sample text", "doc.txt")], None)
        props = result["parameters"]["properties"]
        assert "invoice_number" in props
        assert "amount" in props

    def test_confidence_scores_auto_detected(self):
        """Auto-detection passes confidence_scores=True to refinement when schema has them."""
        from textgleaner.schema_refiner import _build_refinement_system_prompt
        prompt_with = _build_refinement_system_prompt(True)
        prompt_without = _build_refinement_system_prompt(False)
        assert "foo_confidence" in prompt_with
        assert "foo_confidence" not in prompt_without

        # When the existing schema has confidence fields, they should be detected
        assert _detect_confidence_scores(_base_schema(with_confidence=True)) is True
        assert _detect_confidence_scores(_base_schema(with_confidence=False)) is False

    def test_confidence_prompt_sent_when_schema_has_confidence(self):
        """refine_schema passes confidence_scores=True when auto-detected from schema."""
        mock = _mock_client()
        with patch("textgleaner.schema_refiner.LLMClient", return_value=mock):
            refine_schema(_base_schema(with_confidence=True), [("text", "doc.txt")], None)
        # Pass 2 system prompt should mention confidence
        system_msg = mock.chat.call_args_list[1][0][0][0]["content"]
        assert "foo_confidence" in system_msg

    def test_output_file_written(self, tmp_path):
        out = tmp_path / "refined.json"
        mock = _mock_client()
        with patch("textgleaner.schema_refiner.LLMClient", return_value=mock):
            refine_schema(_base_schema(), [("sample text", "doc.txt")], out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert "vendor" in data["parameters"]["properties"]

    def test_raises_on_empty_samples(self):
        mock = _mock_client()
        with patch("textgleaner.schema_refiner.LLMClient", return_value=mock):
            with pytest.raises(ValueError, match="No readable text"):
                refine_schema(_base_schema(), [("", "empty.txt")], None)

    def test_strips_markdown_fences_from_response(self):
        fenced = "```json\n" + json.dumps(_updated_schema()) + "\n```"
        mock = MagicMock()
        mock.chat.return_value = {}
        mock.get_content.side_effect = ["gap analysis", fenced]
        with patch("textgleaner.schema_refiner.LLMClient", return_value=mock):
            result = refine_schema(_base_schema(), [("text", "doc.txt")], None)
        assert result["name"] == "extract_invoice"


class TestPublicRefineSchema:
    def test_schema_as_file_path(self, tmp_path):
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps(_base_schema()))
        sample_file = tmp_path / "sample.txt"
        sample_file.write_text("Invoice #1234 from Acme Corp, total $500")

        mock = _mock_client()
        with patch("textgleaner.schema_refiner.LLMClient", return_value=mock):
            from textgleaner import refine_schema
            result = refine_schema(schema_file, [sample_file])

        assert "vendor" in result["parameters"]["properties"]

    def test_schema_as_dict(self, tmp_path):
        sample_file = tmp_path / "sample.txt"
        sample_file.write_text("Invoice #1234, total $500")

        mock = _mock_client()
        with patch("textgleaner.schema_refiner.LLMClient", return_value=mock):
            from textgleaner import refine_schema
            result = refine_schema(_base_schema(), [sample_file])

        assert result["name"] == "extract_invoice"

    def test_text_instance_as_sample(self):
        mock = _mock_client()
        with patch("textgleaner.schema_refiner.LLMClient", return_value=mock):
            from textgleaner import refine_schema, Text
            result = refine_schema(_base_schema(), Text("sample text", name="section"))

        assert "vendor" in result["parameters"]["properties"]

    def test_output_overwrites_schema_file(self, tmp_path):
        schema_file = tmp_path / "schema.json"
        schema_file.write_text(json.dumps(_base_schema()))
        sample_file = tmp_path / "sample.txt"
        sample_file.write_text("Invoice #1234 from Acme Corp")

        mock = _mock_client()
        with patch("textgleaner.schema_refiner.LLMClient", return_value=mock):
            from textgleaner import refine_schema
            refine_schema(schema_file, [sample_file], output=schema_file)

        data = json.loads(schema_file.read_text())
        assert "vendor" in data["parameters"]["properties"]
