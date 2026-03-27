from __future__ import annotations
from typing import Any


def merge_chunks(chunks: list[dict]) -> dict:
    """
    Merge extraction results from multiple chunks.
    Later chunks overwrite earlier chunks only for fields where the earlier value was null.
    """
    if not chunks:
        return {}
    result: dict[str, Any] = dict(chunks[0])
    for chunk in chunks[1:]:
        for key, value in chunk.items():
            if result.get(key) is None and value is not None:
                result[key] = value
    return result


def null_rate(records: list[dict]) -> dict[str, float]:
    """
    Compute null rate (0.0–1.0) for each field across all records.
    Only considers non-confidence fields.
    """
    if not records:
        return {}
    all_keys = {k for r in records for k in r.keys() if not k.endswith("_confidence")}
    rates = {}
    for key in all_keys:
        null_count = sum(1 for r in records if r.get(key) is None)
        rates[key] = null_count / len(records)
    return rates
