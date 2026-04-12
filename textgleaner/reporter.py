"""Batch output formatting and summary reporting for extract() results.

This module provides three kinds of output:

**Field quality summary** (:func:`summarize`, :func:`build_validation_report`,
:func:`format_validation_report`)
    Given the ``{name: extracted_dict}`` result from :func:`~textgleaner.extract`,
    compute per-field null rates and average confidence scores, then classify
    each field as OK, high-null, always-null, or low-confidence.  Used by
    the ``validate`` CLI command and :func:`~textgleaner.summarize`.

**Tabular output** (:func:`write_csv`, :func:`write_excel`)
    Write extraction results as a spreadsheet — one row per document, one
    column per field.  Nested objects and arrays are JSON-encoded in their
    cell.  :func:`write_excel` requires ``openpyxl``
    (``pip install textgleaner[excel]``).

**Summary CSV** (:func:`write_summary_csv`)
    Write a :func:`summarize` result as a CSV with columns
    ``field``, ``null_rate``, ``avg_confidence``.
"""
from __future__ import annotations
import csv
import json
from pathlib import Path
from typing import Any


def _is_null(value: Any) -> bool:
    """Return ``True`` if *value* should be treated as absent for summary purposes.

    ``None`` and empty lists both count as null.  A non-empty list is not null
    even if every element is ``None``.

    Args:
        value: The field value from an extraction result.

    Returns:
        ``True`` if the value is ``None`` or an empty list.
    """
    if value is None:
        return True
    if isinstance(value, list) and len(value) == 0:
        return True
    return False


def summarize(results: dict) -> dict:
    """Compute per-field null-rate and average confidence from extract() results.

    Scans all top-level fields across every document in *results*.  For each
    field, counts how often it is null (or an empty list) and averages the
    corresponding ``<field>_confidence`` scores where present.

    Only top-level fields are examined — nested objects and array items are
    not recursed into.  Confidence score sibling fields (``<field>_confidence``)
    are excluded from the output; they are consumed but not reported directly.

    Args:
        results: The dict returned by :func:`~textgleaner.extract` for multiple
                 inputs — ``{name: extracted_dict, ...}``.  An empty dict
                 returns ``{}``.

    Returns:
        ``{field_name: {"null_rate": float, "avg_confidence": float | None}, ...}``
        sorted alphabetically by field name.  ``avg_confidence`` is ``None``
        when the schema does not include confidence scoring for that field.

    Example::

        from textgleaner import extract, summarize

        results = extract(["jan.txt", "feb.txt"], schema=schema)
        stats = summarize(results)
        # {"amount": {"null_rate": 0.0, "avg_confidence": 0.95}, ...}
    """
    if not results:
        return {}

    # Collect every field name that appears in at least one document,
    # excluding _confidence siblings (they are read but not reported).
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
            # Read the confidence sibling if present.
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


def build_validation_report(
    summary: dict,
    null_threshold: float = 0.5,
    confidence_threshold: float = 0.5,
) -> dict:
    """Classify each field from a :func:`summarize` result into a validation status.

    A field may accumulate one or more issues:

    - ``"always_null"`` — null_rate == 1.0 (the field was never populated in
      any document in the batch).  This usually means the field name is wrong
      or the document type does not contain that data.
    - ``"high_null"`` — null_rate > *null_threshold* (often missing, but not
      always).  May indicate an optional field or one that only appears in
      certain document variants.
    - ``"low_confidence"`` — avg_confidence < *confidence_threshold* (the field
      is sometimes populated but the model was uncertain about the values).
      Only applied when confidence scores are present.

    A field with no issues gets the implicit status ``"ok"``.

    Args:
        summary: Output of :func:`summarize` — ``{field: {"null_rate", "avg_confidence"}}``
        null_threshold: Null rate above which a field is flagged ``"high_null"``.
            Default 0.5 (more than half the documents had no value).
        confidence_threshold: Average confidence below which a field is flagged
            ``"low_confidence"``.  Default 0.5.  Only applied when
            ``avg_confidence`` is not ``None``.

    Returns:
        A report dict::

            {
                "fields": {
                    "amount": {
                        "null_rate": 0.1,
                        "avg_confidence": 0.92,
                        "issues": []        # empty = OK
                    },
                    "description": {
                        "null_rate": 0.8,
                        "avg_confidence": 0.3,
                        "issues": ["high_null", "low_confidence"]
                    },
                    ...
                },
                "counts": {"ok": 3, "always_null": 1, "high_null": 2, "low_confidence": 1},
                "null_threshold": 0.5,
                "confidence_threshold": 0.5,
            }
    """
    fields: dict = {}
    counts: dict = {"ok": 0, "always_null": 0, "high_null": 0, "low_confidence": 0}

    for field, stats in summary.items():
        null_rate = stats["null_rate"]
        avg_conf = stats["avg_confidence"]
        issues: list[str] = []

        if null_rate == 1.0:
            issues.append("always_null")
        elif null_rate > null_threshold:
            issues.append("high_null")

        if avg_conf is not None and avg_conf < confidence_threshold:
            issues.append("low_confidence")

        fields[field] = {
            "null_rate": null_rate,
            "avg_confidence": avg_conf,
            "issues": issues,
        }
        if not issues:
            counts["ok"] += 1
        else:
            for issue in issues:
                counts[issue] = counts.get(issue, 0) + 1

    return {
        "fields": fields,
        "counts": counts,
        "null_threshold": null_threshold,
        "confidence_threshold": confidence_threshold,
    }


def format_validation_report(report: dict) -> str:
    """Format a :func:`build_validation_report` result as a human-readable table.

    Produces a fixed-width text table with one row per field showing the null
    percentage, average confidence, and status label(s).  A summary line is
    appended at the bottom.

    Args:
        report: The dict returned by :func:`build_validation_report`.

    Returns:
        A multi-line string suitable for printing to stdout.

    Example output::

          Field              Null%  Avg Conf  Status
          ──────────────────────────────────────────
          amount              10%      0.92  OK
          description         80%      0.30  HIGH NULL + LOW CONF
          ...
          5 fields total · 3 OK · 1 high null · 1 low confidence
    """
    fields = report["fields"]
    counts = report["counts"]

    # Column width is based on the longest field name, minimum 20 characters.
    col_w = max((len(f) for f in fields), default=20) + 2
    lines: list[str] = []

    header = f"  {'Field':<{col_w}} {'Null%':>6}  {'Avg Conf':>8}  Status"
    lines.append(header)
    lines.append("  " + "─" * (len(header) - 2))

    for field, info in fields.items():
        null_pct = f"{info['null_rate'] * 100:.0f}%"
        conf = f"{info['avg_confidence']:.2f}" if info["avg_confidence"] is not None else "—"
        issues = info["issues"]
        if not issues:
            status = "OK"
        else:
            label_map = {
                "always_null": "ALWAYS NULL",
                "high_null": "HIGH NULL",
                "low_confidence": "LOW CONF",
            }
            status = " + ".join(label_map[i] for i in issues)
        lines.append(f"  {field:<{col_w}} {null_pct:>6}  {conf:>8}  {status}")

    lines.append("")
    n_ok = counts.get("ok", 0)
    total = len(fields)
    parts = [f"{total} fields total", f"{n_ok} OK"]
    if counts.get("always_null"):
        parts.append(f"{counts['always_null']} always null")
    if counts.get("high_null"):
        parts.append(f"{counts['high_null']} high null")
    if counts.get("low_confidence"):
        parts.append(f"{counts['low_confidence']} low confidence")
    lines.append("  " + " · ".join(parts))

    return "\n".join(lines)


def write_csv(results: dict, path: Path) -> None:
    """Write extraction results to a CSV file.

    Produces a CSV with one row per document.  The first column is
    ``filename`` (the dict key from *results*), followed by one column per
    extracted field in the order they first appear across all documents.

    Nested objects and arrays are JSON-encoded in their cell so the CSV
    remains flat.  If *results* is empty, a header-only file is written.

    Args:
        results: ``{name: extracted_dict, ...}`` as returned by
            :func:`~textgleaner.extract` for multiple inputs.
        path: Destination CSV file path.  The file is created or overwritten.
    """
    if not results:
        path.write_text("filename\n", encoding="utf-8")
        return

    # Collect field names in first-seen order across all documents.
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
                # Nested values are JSON-encoded so the cell contains a valid
                # string (e.g. '[{"date": "01/15/24", ...}]' for an array).
                row[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
            writer.writerow(row)


def write_summary_csv(summary: dict, path: Path) -> None:
    """Write a :func:`summarize` result to a CSV file.

    Produces a CSV with columns ``field``, ``null_rate``, ``avg_confidence``,
    one row per field.

    Args:
        summary: Output of :func:`summarize`.
        path: Destination CSV file path.  The file is created or overwritten.
    """
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["field", "null_rate", "avg_confidence"])
        writer.writeheader()
        for field, stats in summary.items():
            writer.writerow({"field": field, **stats})


def write_excel(results: dict, path: Path) -> None:
    """Write extraction results to an Excel (.xlsx) workbook.

    Produces a single worksheet named "Extracted Data" with one row per
    document and one column per field.  The header row is bold.  Nested
    objects and arrays are JSON-encoded in their cell.

    Requires the ``openpyxl`` package::

        pip install textgleaner[excel]

    Args:
        results: ``{name: extracted_dict, ...}`` as returned by
            :func:`~textgleaner.extract` for multiple inputs.
        path: Destination ``.xlsx`` file path.

    Raises:
        ImportError: If ``openpyxl`` is not installed.
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

    # Collect field names in first-seen order.
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
