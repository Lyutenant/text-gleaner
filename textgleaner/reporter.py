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


def build_validation_report(
    summary: dict,
    null_threshold: float = 0.5,
    confidence_threshold: float = 0.5,
) -> dict:
    """Classify each field from a :func:`summarize` result into a validation status.

    A field accumulates issues based on these thresholds:

    - ``always_null``    — null_rate == 1.0 (field never populated)
    - ``high_null``      — null_rate > null_threshold (often missing)
    - ``low_confidence`` — avg_confidence < confidence_threshold (present but uncertain)

    Args:
        summary: Output of :func:`summarize`.
        null_threshold: null_rate above which a field is flagged ``high_null``.
        confidence_threshold: avg_confidence below which a field is flagged
            ``low_confidence``. Only applied when confidence scores are present.

    Returns:
        ``{"fields": {field: {"null_rate", "avg_confidence", "issues"}}, "counts": {...}}``
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
    """Format a :func:`build_validation_report` result as a human-readable table."""
    fields = report["fields"]
    counts = report["counts"]

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
    n_issues = sum(v for k, v in counts.items() if k != "ok")
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
