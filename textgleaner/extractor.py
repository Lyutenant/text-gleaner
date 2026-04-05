from __future__ import annotations
import json
import logging
import re
from pathlib import Path  # used only for output_path
from typing import Any, Callable

import httpx

from .config import ExtractionConfig
from .llm_client import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TOOL_CALL = """\
You are a precise data extraction assistant. You MUST respond by calling the provided tool function \
with the extracted values. Do NOT write any plain text — only make the tool call.

Rules:
- You MUST call the tool. This is mandatory. No text response is acceptable.
- Extract ONLY information explicitly present in the document text.
- Never infer, guess, or hallucinate values.
- Use null for any field whose value is not present in the document.
- Confidence score meanings:
  - 1.0 = value is explicitly stated verbatim
  - 0.7 = value is clearly implied
  - 0.4 = value is inferred / uncertain
  - 0.0 = value not found (field will be null)
"""

SYSTEM_PROMPT_STRUCTURED = """\
You are a precise data extraction assistant. Extract information from the document text and return \
it as JSON matching the provided schema exactly.

Rules:
- Extract ONLY information explicitly present in the document text.
- Never infer, guess, or hallucinate values.
- Use null for any field whose value is not present in the document.
- Confidence score meanings:
  - 1.0 = value is explicitly stated verbatim
  - 0.7 = value is clearly implied
  - 0.4 = value is inferred / uncertain
  - 0.0 = value not found (field will be null)
"""


def _check_size(text: str, name: str, max_chars: int) -> None:
    if max_chars > 0 and len(text) > max_chars:
        raise ValueError(
            f"Input '{name}' exceeds max_chars limit "
            f"({len(text):,} > {max_chars:,}). "
            f"Split the file or increase max_chars."
        )


def _extract_one_tool_call(client: LLMClient, schema: dict, text: str, filename: str) -> dict:
    """Extract using forced tool call (tool_choice). Falls back to content JSON if the
    model ignores tool_choice and returns JSON in the content field instead."""
    tool_def = {
        "type": "function",
        "function": {
            "name": schema["name"],
            "description": schema.get("description", ""),
            "parameters": schema["parameters"],
        },
    }
    tool_choice = {"type": "function", "function": {"name": schema["name"]}}
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_TOOL_CALL},
        {"role": "user", "content": f"Document text:\n\n{text}"},
    ]
    try:
        response = client.chat(messages, tools=[tool_def], tool_choice=tool_choice)
        return client.get_tool_arguments(response)
    except Exception as e:
        logger.warning("filename=%s error=%s", filename, e)
        raise


def _extract_one_structured(client: LLMClient, schema: dict, text: str, filename: str) -> dict:
    """Extract using response_format / json_schema (grammar-constrained decoding).
    Works with models that support structured outputs but handle tool_choice poorly."""
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": schema["name"],
            "schema": schema["parameters"],
            "strict": True,
        },
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_STRUCTURED},
        {"role": "user", "content": f"Document text:\n\n{text}"},
    ]

    def _parse(raw: str) -> dict:
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
        return json.loads(raw)

    try:
        response = client.chat(messages, response_format=response_format)
        content = client.get_content(response).strip()
        if not content:
            # Some models (e.g. Qwen3) occasionally return empty content with
            # response_format. Retry once before giving up.
            logger.warning("filename=%s structured_output returned empty content, retrying", filename)
            response = client.chat(messages, response_format=response_format)
            content = client.get_content(response).strip()
        return _parse(content)
    except Exception as e:
        logger.warning("filename=%s error=%s", filename, e)
        raise


RETRY_CONFIDENCE_THRESHOLD = 0.4


def _build_retry_schema(schema: dict, fields: list[str]) -> dict:
    """Return a copy of schema containing only *fields* and their _confidence siblings."""
    orig_props = schema["parameters"].get("properties", {})
    props: dict = {}
    for f in fields:
        if f in orig_props:
            props[f] = orig_props[f]
        conf_key = f"{f}_confidence"
        if conf_key in orig_props:
            props[conf_key] = orig_props[conf_key]
    return {
        "name": schema["name"],
        "description": schema.get("description", ""),
        "parameters": {"type": "object", "properties": props},
    }


def _retry_low_confidence(
    client: LLMClient,
    schema: dict,
    text: str,
    filename: str,
    result: dict,
    method: str,
) -> dict:
    """Re-extract fields whose confidence score is ≤ RETRY_CONFIDENCE_THRESHOLD.

    Sends a second, narrowed extraction call covering only the weak fields.
    A field is updated in the result only when the retry returns a strictly
    higher confidence score, so a failed retry never makes things worse.

    Fields without a ``_confidence`` sibling (i.e. confidence_scores disabled)
    are silently skipped.
    """
    low_conf_fields: list[str] = []
    for key in result:
        if key.endswith("_confidence"):
            continue
        conf = result.get(f"{key}_confidence")
        if conf is not None and conf <= RETRY_CONFIDENCE_THRESHOLD:
            low_conf_fields.append(key)

    if not low_conf_fields:
        return result

    logger.info(
        "filename=%s confidence_retry: %d field(s) at or below %.1f threshold: %s",
        filename, len(low_conf_fields), RETRY_CONFIDENCE_THRESHOLD, ", ".join(low_conf_fields),
    )

    retry_schema = _build_retry_schema(schema, low_conf_fields)
    try:
        if method == "structured_output":
            retry_result = _extract_one_structured(client, retry_schema, text, filename)
        elif method == "auto":
            retry_result = _extract_one_auto(client, retry_schema, text, filename)
        else:
            retry_result = _extract_one_tool_call(client, retry_schema, text, filename)
    except Exception as e:
        logger.warning(
            "filename=%s confidence_retry failed (%s) — keeping original result", filename, e,
        )
        return result

    updated = dict(result)
    improved = 0
    for field in low_conf_fields:
        orig_conf = result.get(f"{field}_confidence") or 0.0
        retry_conf = retry_result.get(f"{field}_confidence")
        if retry_conf is not None and retry_conf > orig_conf:
            updated[field] = retry_result[field]
            updated[f"{field}_confidence"] = retry_conf
            improved += 1

    logger.info(
        "filename=%s confidence_retry: improved %d/%d field(s)",
        filename, improved, len(low_conf_fields),
    )
    return updated


def _extract_one_auto(client: LLMClient, schema: dict, text: str, filename: str) -> dict:
    """Try tool_call first; fall back to structured_output if the model or server
    cannot handle the tool call.

    Fallback is triggered by:
    - ValueError / JSONDecodeError — model ignored tool_choice and returned
      unparseable output
    - HTTP 400 / 422 — server rejected the tools payload (model doesn't support
      tool calls)

    All other exceptions (timeouts, HTTP 5xx, etc.) are re-raised immediately
    since they would fail the same way on the structured_output path.
    """
    try:
        return _extract_one_tool_call(client, schema, text, filename)
    except (ValueError, json.JSONDecodeError) as e:
        logger.warning(
            "filename=%s tool_call produced no usable output (%s) — retrying with structured_output",
            filename, type(e).__name__,
        )
    except httpx.HTTPStatusError as e:
        if e.response.status_code in (400, 422):
            logger.warning(
                "filename=%s HTTP %d on tool_call request — retrying with structured_output",
                filename, e.response.status_code,
            )
        else:
            raise
    return _extract_one_structured(client, schema, text, filename)


def extract(
    inputs: list[tuple[str, str]],
    schema: dict,
    output_path: Path | None,
    single: bool,
    *,
    max_chars: int | None = None,
    extraction_method: str | None = None,
    confidence_retry: bool | None = None,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: int | None = None,
    model_profile: str | None = None,
    on_result: Callable[[str, dict], None] | None = None,
) -> dict:
    cfg = ExtractionConfig()
    effective_max = max_chars if max_chars is not None else cfg.max_chars
    effective_method = extraction_method or cfg.extraction_method
    effective_retry = confidence_retry if confidence_retry is not None else cfg.confidence_retry

    client = LLMClient(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        model_profile=model_profile,
    )

    results: dict[str, Any] = {}

    for text, name in inputs:
        _check_size(text, name, effective_max)
        logger.info("Extracting from %s (%d chars) method=%s", name, len(text), effective_method)
        if effective_method == "structured_output":
            data = _extract_one_structured(client, schema, text, name)
        elif effective_method == "auto":
            data = _extract_one_auto(client, schema, text, name)
        else:  # tool_call
            data = _extract_one_tool_call(client, schema, text, name)
        if effective_retry:
            data = _retry_low_confidence(client, schema, text, name, data, effective_method)
        results[name] = data
        if on_result is not None:
            on_result(name, data)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        first_name = inputs[0][1]
        payload = results[first_name] if single else results
        with output_path.open("w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        logger.info("Wrote %s", output_path)

    if single:
        return results[inputs[0][1]]
    return results
