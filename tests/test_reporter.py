from __future__ import annotations
import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from textgleaner.reporter import summarize, write_csv, write_summary_csv


class TestSummarize:
    def _results(self):
        return {
            "doc1.txt": {
                "account": "111",
                "account_confidence": 1.0,
                "amount": None,
                "amount_confidence": 0.0,
                "items": [{"name": "x"}],
                "items_confidence": 0.9,
            },
            "doc2.txt": {
                "account": "222",
                "account_confidence": 0.7,
                "amount": "50.00",
                "amount_confidence": 1.0,
                "items": [],          # empty list → counts as null
                "items_confidence": 0.0,
            },
        }

    def test_null_rate_scalar(self):
        summary = summarize(self._results())
        # "amount": null in doc1, present in doc2 → 50% null
        assert summary["amount"]["null_rate"] == 0.5

    def test_null_rate_zero(self):
        summary = summarize(self._results())
        assert summary["account"]["null_rate"] == 0.0

    def test_empty_list_counts_as_null(self):
        summary = summarize(self._results())
        # items: one non-empty, one empty → 50% null
        assert summary["items"]["null_rate"] == 0.5

    def test_avg_confidence(self):
        summary = summarize(self._results())
        assert summary["account"]["avg_confidence"] == pytest.approx((1.0 + 0.7) / 2)

    def test_confidence_fields_excluded_from_keys(self):
        summary = summarize(self._results())
        assert not any(k.endswith("_confidence") for k in summary)

    def test_fields_sorted_alphabetically(self):
        summary = summarize(self._results())
        assert list(summary.keys()) == sorted(summary.keys())

    def test_empty_results(self):
        assert summarize({}) == {}

    def test_no_confidence_fields(self):
        results = {"a.txt": {"value": "x"}, "b.txt": {"value": None}}
        summary = summarize(results)
        assert summary["value"]["avg_confidence"] is None
        assert summary["value"]["null_rate"] == 0.5


class TestWriteCsv:
    def test_writes_one_row_per_doc(self, tmp_path):
        results = {
            "a.txt": {"field1": "hello", "field2": 42},
            "b.txt": {"field1": None, "field2": 99},
        }
        out = tmp_path / "out.csv"
        write_csv(results, out)

        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        assert len(rows) == 2
        assert rows[0]["filename"] == "a.txt"
        assert rows[1]["filename"] == "b.txt"

    def test_nested_values_json_encoded(self, tmp_path):
        results = {"a.txt": {"items": [{"name": "x", "val": 1}]}}
        out = tmp_path / "out.csv"
        write_csv(results, out)

        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        parsed = json.loads(rows[0]["items"])
        assert parsed == [{"name": "x", "val": 1}]

    def test_empty_results(self, tmp_path):
        out = tmp_path / "out.csv"
        write_csv({}, out)
        assert "filename" in out.read_text()


class TestWriteSummaryCsv:
    def test_writes_all_fields(self, tmp_path):
        summary = {
            "amount": {"null_rate": 0.5, "avg_confidence": 0.75},
            "account": {"null_rate": 0.0, "avg_confidence": 0.9},
        }
        out = tmp_path / "summary.csv"
        write_summary_csv(summary, out)

        rows = list(csv.DictReader(out.read_text(encoding="utf-8").splitlines()))
        assert len(rows) == 2
        fields = {r["field"] for r in rows}
        assert fields == {"amount", "account"}


class TestPublicSummarize:
    def test_returns_summary_dict(self):
        from textgleaner import summarize
        results = {"a.txt": {"x": "1", "x_confidence": 1.0}, "b.txt": {"x": None, "x_confidence": 0.0}}
        summary = summarize(results)
        assert "x" in summary
        assert summary["x"]["null_rate"] == 0.5

    def test_writes_csv_when_output_given(self, tmp_path):
        from textgleaner import summarize
        results = {"a.txt": {"x": "1"}, "b.txt": {"x": None}}
        out = tmp_path / "summary.csv"
        summarize(results, output=out)
        assert out.exists()
        rows = list(csv.DictReader(out.read_text().splitlines()))
        assert rows[0]["field"] == "x"


class TestExtractCsvOutput:
    def _schema(self):
        return {
            "name": "extract_data",
            "description": "Test",
            "parameters": {"type": "object", "properties": {
                "value": {"type": ["string", "null"], "description": "Value"},
            }},
        }

    def test_csv_output_written(self, tmp_path):
        from unittest.mock import MagicMock, patch
        from textgleaner import extract, Text

        mock_client = MagicMock()
        mock_client.chat.return_value = {}
        mock_client.get_tool_arguments.side_effect = [{"value": "a"}, {"value": "b"}]

        out = tmp_path / "results.csv"
        with patch("textgleaner.extractor.LLMClient", return_value=mock_client):
            result = extract(
                [Text("doc a", name="a"), Text("doc b", name="b")],
                schema=self._schema(),
                output=out,
            )

        assert out.exists()
        rows = list(csv.DictReader(out.read_text().splitlines()))
        assert len(rows) == 2
        assert {r["filename"] for r in rows} == {"a", "b"}
        # extract() should still return the normal results dict
        assert result == {"a": {"value": "a"}, "b": {"value": "b"}}
