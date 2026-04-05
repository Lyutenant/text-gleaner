"""Batch output formatting and summary reporting for extract() results."""
from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Any


def _is_null(value: Any) -> bool:
    """True if a field value counts as absent for summary purposes."""
    if value is None:
        return True
    if isinstance(value, list) and len(value) == 0:
        return True
    return False


def summarize(results: dict) -> dict:
    """Compute per-field null-rate and average confidence from extract() results.

    Only top-level fields are examined. For array fields, an empty list counts
    as null. Confidence scores are read from ``<field>_confidence`` siblings.

    Args:
        results: The dict returned by :func:`textgleaner.extract` for multiple
                 inputs — ``{name: extracted_dict, ...}``.

    Returns:
        ``{field_name: {"null_rate": float, "avg_confidence": float | None}, ...}``
        sorted alphabetically by field name.
    """
    if not results:
        return {}

    all_fields: set[str] = set()
    for doc in results.values():
        all_fields.update(k for k in doc if not k.endswith("_confidence"))

    n = len(results)
    summary: dict = {}
    for field in sorted(all_fields):
        null_count = 0
        confidence_vals: list[float] = []

        for doc in results.values():
            value = doc.get(field)
            if _is_null(value):
                null_count += 1
            conf = doc.get(f"{field}_confidence")
            if conf is not None:
                confidence_vals.append(float(conf))

        summary[field] = {
            "null_rate": round(null_count / n, 4),
            "avg_confidence": (
                round(sum(confidence_vals) / len(confidence_vals), 4)
                if confidence_vals else None
            ),
        }

    return summary


def write_csv(results: dict, path: Path) -> None:
    """Write extract() results to a CSV file.

    One row per document. Nested objects and arrays are JSON-encoded in their cell.
    """
    if not results:
        path.write_text("filename\n", encoding="utf-8")
        return

    all_fields: list[str] = []
    seen: set[str] = set()
    for doc in results.values():
        for k in doc:
            if k not in seen:
                all_fields.append(k)
                seen.add(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename"] + all_fields, extrasaction="ignore"
        )
        writer.writeheader()
        for filename, doc in results.items():
            row: dict = {"filename": filename}
            for k, v in doc.items():
                row[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
            writer.writerow(row)


def write_summary_csv(summary: dict, path: Path) -> None:
    """Write a :func:`summarize` result to a CSV file."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["field", "null_rate", "avg_confidence"])
        writer.writeheader()
        for field, stats in summary.items():
            writer.writerow({"field": field, **stats})


def write_excel(results: dict, path: Path) -> None:
    """Write extract() results to an Excel (.xlsx) file.

    Requires openpyxl::

        pip install textgleaner[excel]
    """
    try:
        import openpyxl
        from openpyxl.styles import Font
    except ImportError:
        raise ImportError(
            "Excel output requires openpyxl. "
            "Install it with: pip install textgleaner[excel]"
        ) from None

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Extracted Data"

    if not results:
        wb.save(path)
        return

    all_fields: list[str] = []
    seen: set[str] = set()
    for doc in results.values():
        for k in doc:
            if k not in seen:
                all_fields.append(k)
                seen.add(k)

    headers = ["filename"] + all_fields
    ws.append(headers)
    for cell in ws[1]:
        cell.font = Font(bold=True)

    for filename, doc in results.items():
        row = [filename]
        for field in all_fields:
            v = doc.get(field)
            row.append(json.dumps(v) if isinstance(v, (dict, list)) else v)
        ws.append(row)

    wb.save(path)
