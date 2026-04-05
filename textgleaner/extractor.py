from __future__ import annotations
import json
import logging
import re
from pathlib import Path  # used only for output_path
from typing import Any

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
    try:
        response = client.chat(messages, response_format=response_format)
        content = client.get_content(response).strip()
        # Strip markdown code fences if present
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content).strip()
        return json.loads(content)
    except Exception as e:
        logger.warning("filename=%s error=%s", filename, e)
        raise


def extract(
    inputs: list[tuple[str, str]],
    schema: dict,
    output_path: Path | None,
    single: bool,
    *,
    max_chars: int | None = None,
    extraction_method: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: int | None = None,
) -> dict:
    cfg = ExtractionConfig()
    effective_max = max_chars if max_chars is not None else cfg.max_chars
    effective_method = extraction_method or cfg.extraction_method

    client = LLMClient(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )

    results: dict[str, Any] = {}

    for text, name in inputs:
        _check_size(text, name, effective_max)
        logger.info("Extracting from %s (%d chars) method=%s", name, len(text), effective_method)
        if effective_method == "structured_output":
            data = _extract_one_structured(client, schema, text, name)
        else:  # tool_call or auto (auto uses tool_call with content fallback)
            data = _extract_one_tool_call(client, schema, text, name)
        results[name] = data

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
