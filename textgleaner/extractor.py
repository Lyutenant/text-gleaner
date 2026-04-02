from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any

from .config import ExtractionConfig
from .llm_client import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
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


def _check_size(text: str, path: Path, max_chars: int) -> None:
    if max_chars > 0 and len(text) > max_chars:
        raise ValueError(
            f"Input '{path.name}' exceeds max_chars limit "
            f"({len(text):,} > {max_chars:,}). "
            f"Split the file or increase max_chars."
        )


def _extract_one(client: LLMClient, schema: dict, text: str, filename: str) -> dict:
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
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Document text:\n\n{text}"},
    ]
    try:
        response = client.chat(messages, tools=[tool_def], tool_choice=tool_choice)
        return client.get_tool_arguments(response)
    except Exception as e:
        logger.warning("filename=%s error=%s", filename, e)
        raise


def extract(
    input_paths: list[Path],
    schema: dict,
    output_path: Path | None,
    single: bool,
    *,
    max_chars: int | None = None,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: int | None = None,
) -> dict:
    effective_max = max_chars if max_chars is not None else ExtractionConfig().max_chars
    client = LLMClient(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )

    results: dict[str, Any] = {}

    for path in input_paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        _check_size(text, path, effective_max)
        logger.info("Extracting from %s (%d chars)", path.name, len(text))
        data = _extract_one(client, schema, text, path.name)
        results[path.name] = data

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = results[input_paths[0].name] if single else results
        with output_path.open("w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        logger.info("Wrote %s", output_path)

    if single:
        return results[input_paths[0].name]
    return results
