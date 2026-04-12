"""Phase 2 extraction logic for textgleaner.

This module implements three extraction strategies and the retry logic:

``tool_call`` (default)
    The schema is sent as a tool definition with ``tool_choice`` set to force
    the model to call it.  The model responds with
    ``tool_calls[0].function.arguments`` — a JSON string containing the
    extracted data.  This is the fastest path and works well with large,
    instruction-following models (Qwen3, GPT-4o, Claude, etc.).

    Fallback: some Ollama models ignore ``tool_choice`` and embed the JSON
    directly in the ``content`` field.  :func:`LLMClient.get_tool_arguments`
    detects this and parses the content instead, so callers are unaffected.

``structured_output``
    The schema is sent as ``response_format: {type: json_schema}`` so the
    inference engine applies grammar-constrained token sampling — the model
    physically cannot produce output that violates the schema.  More reliable
    on small or weak models, but much slower (grammar compilation overhead).

    Caveat: Qwen3 with ``think: false`` occasionally returns empty content
    via this path.  :func:`_extract_one_structured` retries once automatically.

``auto``
    Tries ``tool_call`` first.  Falls back to ``structured_output`` on
    :exc:`ValueError`, :exc:`json.JSONDecodeError`, or HTTP 400/422 (server
    rejected the tools payload).  Timeouts and HTTP 5xx are always re-raised
    immediately since those failures would occur on either path.

The public entry point :func:`extract` also supports optional
**confidence retry**: after the initial extraction, any field whose
``_confidence`` score is ≤ 0.4 triggers a second, narrowed call that focuses
only on those weak fields.  A field is updated only when the retry returns a
strictly higher confidence, so a failed retry is harmlessly discarded.
"""
from __future__ import annotations
import json
import logging
import re
from pathlib import Path  # used only for output_path
from typing import Any, Callable

import httpx

from .config import ExtractionConfig
from .llm_client import LLMClient, make_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

# Used for tool_call extraction.  Emphasises that the tool MUST be called.
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

# Used for structured_output extraction.  No tool call instruction needed
# because grammar-constrained decoding enforces the output shape directly.
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

# ---------------------------------------------------------------------------
# Size guard
# ---------------------------------------------------------------------------

def _check_size(text: str, name: str, max_chars: int) -> None:
    """Raise :exc:`ValueError` if *text* exceeds the configured character limit.

    The check is skipped entirely when *max_chars* is 0 (no limit).  The guard
    exists to prevent accidentally sending enormous files to the LLM, which
    would either exceed the model's context window or take a very long time.

    Args:
        text: The document text to check.
        name: Human-readable name for the document (used in the error message).
        max_chars: Maximum allowed characters.  0 means no limit.

    Raises:
        ValueError: If ``len(text) > max_chars > 0``.
    """
    if max_chars > 0 and len(text) > max_chars:
        raise ValueError(
            f"Input '{name}' exceeds max_chars limit "
            f"({len(text):,} > {max_chars:,}). "
            f"Split the file or increase max_chars."
        )


# ---------------------------------------------------------------------------
# Extraction strategies
# ---------------------------------------------------------------------------

def _extract_one_tool_call(client: LLMClient, schema: dict, text: str, filename: str) -> dict:
    """Extract structured data by forcing the model to make a tool call.

    Builds a tool definition from *schema* and sends it with
    ``tool_choice`` set to the tool's name, which instructs the model to
    respond exclusively by calling that tool.

    The tool definition format (OpenAI-compatible)::

        {
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema["description"],
                "parameters": schema["parameters"]   # JSON Schema object
            }
        }

    :meth:`~llm_client.LLMClient.get_tool_arguments` is called on the response
    and handles the case where the model ignores ``tool_choice`` and returns
    JSON in the ``content`` field instead.

    Args:
        client: An LLM client instance (either :class:`~llm_client.LLMClient`
            or :class:`~llm_client.ClaudeAPIClient`).
        schema: The extraction schema dict (must have ``"name"`` and
            ``"parameters"`` keys).
        text: The document text to extract from.
        filename: Label for log messages and error reporting.

    Returns:
        Extracted data dict as returned by the model.

    Raises:
        ValueError: If the model returns no tool call and no parseable content.
        json.JSONDecodeError: If the model's JSON output is malformed.
    """
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
    """Extract structured data using grammar-constrained decoding.

    Sends the schema as ``response_format: {type: json_schema}`` so the
    inference engine (llama.cpp / Ollama) compiles the schema into a grammar
    and applies it during token sampling.  The model cannot generate output
    that violates the schema.

    For Claude the ``response_format`` is translated to ``output_config`` by
    :class:`~llm_client.ClaudeAPIClient`.

    Retries once if the model returns empty content — this occasionally
    happens with Qwen3 when ``think: false`` is set and grammar constraints
    are very complex.

    Args:
        client: An LLM client instance.
        schema: The extraction schema dict.
        text: The document text to extract from.
        filename: Label for log messages.

    Returns:
        Extracted data dict parsed from the model's JSON response.

    Raises:
        json.JSONDecodeError: If the model output cannot be parsed as JSON.
        ValueError: If the model returns empty content on both attempts.
    """
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
        """Strip optional markdown fences and parse as JSON."""
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw).strip()
        return json.loads(raw)

    try:
        response = client.chat(messages, response_format=response_format)
        content = client.get_content(response).strip()
        if not content:
            # Some models (e.g. Qwen3) occasionally return empty content with
            # response_format.  Retry once before giving up.
            logger.warning("filename=%s structured_output returned empty content, retrying", filename)
            response = client.chat(messages, response_format=response_format)
            content = client.get_content(response).strip()
        return _parse(content)
    except Exception as e:
        logger.warning("filename=%s error=%s", filename, e)
        raise


# ---------------------------------------------------------------------------
# Confidence retry
# ---------------------------------------------------------------------------

# Fields with a _confidence score at or below this threshold are considered
# "not found" or highly uncertain and are candidates for a retry call.
RETRY_CONFIDENCE_THRESHOLD = 0.4


def _build_retry_schema(schema: dict, fields: list[str]) -> dict:
    """Return a narrowed copy of *schema* containing only *fields* and their confidence siblings.

    Used by :func:`_retry_low_confidence` to build a focused schema for the
    second extraction call.  A narrower schema means the model can concentrate
    entirely on the weak fields rather than re-extracting everything.

    Args:
        schema: The full extraction schema dict.
        fields: List of data field names to include (confidence siblings are
            added automatically if present in the original schema).

    Returns:
        A new schema dict with only the specified fields (and their
        ``<field>_confidence`` siblings if the original has them).
    """
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
    """Re-extract fields whose confidence score is at or below the threshold.

    After the initial extraction, this function scans the result for fields
    with ``_confidence`` scores ≤ :data:`RETRY_CONFIDENCE_THRESHOLD` (0.4 —
    meaning the model was uncertain or the field was not found).  For those
    fields a second LLM call is made using a narrowed schema so the model
    can focus exclusively on finding them.

    **Merge rule**: a field is updated in the result only when the retry
    returns a *strictly higher* confidence score.  This guarantees the retry
    can never make things worse — if the retry also fails or is equally
    uncertain, the original value is kept.

    Fields without a ``_confidence`` sibling (i.e. confidence_scores disabled
    in config) are silently skipped because there is no score to evaluate.

    If the retry call itself raises an exception (network error, parse error,
    etc.), the original result is returned unchanged and a warning is logged.

    Args:
        client: An LLM client instance.
        schema: The full extraction schema (used to build the narrowed schema).
        text: The original document text (same input as the initial call).
        filename: Label for log messages.
        result: The initial extraction result dict.
        method: The extraction method to use for the retry call (``"tool_call"``,
            ``"structured_output"``, or ``"auto"``).

    Returns:
        The (possibly updated) result dict.
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


# ---------------------------------------------------------------------------
# Auto mode (try tool_call, fall back to structured_output)
# ---------------------------------------------------------------------------

def _extract_one_auto(client: LLMClient, schema: dict, text: str, filename: str) -> dict:
    """Try tool_call extraction; fall back to structured_output on failure.

    This is the ``"auto"`` extraction method.  It tries the faster
    :func:`_extract_one_tool_call` path first and only switches to the slower
    :func:`_extract_one_structured` path when the tool_call path fails in a
    way that suggests the model or server does not support it.

    Fallback is triggered by:

    - :exc:`ValueError` or :exc:`json.JSONDecodeError` — the model returned
      something that could not be parsed as the expected JSON (ignored
      tool_choice, returned garbage, etc.).
    - HTTP 400 or 422 — the server rejected the ``tools`` payload outright
      (the model does not support tool calls at all).

    All other exceptions (timeouts, HTTP 5xx, connection errors) are
    **re-raised immediately** because they indicate infrastructure problems
    that would fail on the ``structured_output`` path too.

    Args:
        client: An LLM client instance.
        schema: The extraction schema dict.
        text: The document text to extract from.
        filename: Label for log messages.

    Returns:
        Extracted data dict from whichever path succeeded.

    Raises:
        Any exception that is not in the fallback-trigger list above.
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


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract(
    inputs: list[tuple[str, str]],
    schema: dict,
    output_path: Path | None,
    single: bool,
    *,
    max_chars: int | None = None,
    extraction_method: str | None = None,
    confidence_retry: bool | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: int | None = None,
    model_profile: str | None = None,
    on_result: Callable[[str, dict], None] | None = None,
) -> dict:
    """Extract structured data from one or more pre-loaded text inputs.

    This is the internal entry point called by the public
    :func:`textgleaner.extract` function after it has resolved file paths,
    loaded text, and merged config.  It is not part of the public API.

    For each input the function:

    1. Checks the character count against *max_chars*.
    2. Calls the appropriate extraction function based on *extraction_method*.
    3. Optionally runs the confidence retry pass.
    4. Stores the result and calls *on_result* if provided.

    After all inputs are processed, writes the combined result to
    *output_path* if given, then returns the result.

    Args:
        inputs: List of ``(text, name)`` tuples.  *text* is the document
            content; *name* is used as the dict key in the result and in log
            messages.
        schema: The extraction schema dict.
        output_path: Optional path to write the result JSON.  For a single
            input the flat dict is written; for multiple inputs the
            ``{name: dict}`` mapping is written.
        single: If ``True`` the caller passed a single input, so the return
            value should be the flat dict rather than the ``{name: dict}``
            mapping.
        max_chars: Per-input character limit.  Falls back to
            :class:`~config.ExtractionConfig` default (200 000).
        extraction_method: ``"tool_call"``, ``"structured_output"``, or
            ``"auto"``.  Falls back to ``ExtractionConfig`` default.
        confidence_retry: Whether to run the confidence retry pass.  Falls
            back to ``ExtractionConfig`` default.
        provider: LLM backend — ``"ollama"`` or ``"claude"``.  Passed to
            :func:`~llm_client.make_client`.
        base_url, model, api_key, temperature, max_tokens, timeout,
        model_profile: Passed to :func:`~llm_client.make_client`.
        on_result: Optional callback invoked immediately after each input is
            extracted — useful for streaming progress or writing per-file
            output before the full batch completes.  Called as
            ``on_result(name, result_dict)``.

    Returns:
        For a single input: the flat extracted dict.
        For multiple inputs: ``{name: dict, ...}``.
    """
    cfg = ExtractionConfig()
    effective_max = max_chars if max_chars is not None else cfg.max_chars
    effective_method = extraction_method or cfg.extraction_method
    effective_retry = confidence_retry if confidence_retry is not None else cfg.confidence_retry

    client = make_client(
        provider=provider,
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
        else:  # tool_call (default)
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
