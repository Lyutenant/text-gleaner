from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any

from .config import ExtractionConfig
from .llm_client import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a data extraction schema designer. Given sample document text and a description, \
generate a single JSON object that is a valid OpenAI-compatible tool/function definition for \
structured data extraction.

Requirements:
- Return ONLY valid JSON — no markdown fences, no commentary.
- Top-level keys: "name" (snake_case identifier), "description" (document type description), \
"parameters" (JSON Schema object with "type": "object" and "properties").
- Every data field in "properties" must have:
  - "type": use ["string", "null"] for optional string fields, "number" for numerics, etc.
  - "description": where in the document to find this value
{confidence_instruction}
- Base fields only on information present in the description and sample documents.
- Do not hallucinate field names.
"""

CONFIDENCE_INSTRUCTION = """\
- For each data field "foo", add a sibling field "foo_confidence" with:
  - "type": "number"
  - "description": "Confidence 0-1: 1.0=verbatim, 0.7=implied, 0.4=inferred, 0.0=not found"
"""

RETRY_PROMPT = """\
The previous response was not valid JSON. Here is the error: {error}

Please return ONLY a valid JSON object — no markdown, no explanation. Try again.
"""


def _build_system_prompt(confidence_scores: bool) -> str:
    ci = CONFIDENCE_INSTRUCTION if confidence_scores else ""
    return SYSTEM_PROMPT.format(confidence_instruction=ci)


def _parse_schema_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)


def _validate_schema(schema: dict) -> None:
    for key in ("name", "description", "parameters"):
        if key not in schema:
            raise ValueError(f"Schema missing required key: '{key}'")
    if "properties" not in schema["parameters"]:
        raise ValueError("Schema 'parameters' missing 'properties'")


def generate_schema(
    sample_paths: list[Path],
    description: str,
    output_path: Path | None,
    *,
    confidence_scores: bool | None = None,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: int | None = None,
) -> dict:
    if confidence_scores is None:
        confidence_scores = ExtractionConfig().confidence_scores

    client = LLMClient(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )

    snippets: list[str] = []
    for path in sample_paths:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            snippets.append(f"=== {path.name} ===\n{text}")
        else:
            logger.warning("filename=%s error=empty_file", path.name)

    if not snippets:
        raise ValueError("No readable text found in any sample file.")

    sample_text = "\n\n".join(snippets)
    user_message = (
        f"Document description:\n{description}\n\n"
        f"Sample document text:\n{sample_text}\n\n"
        "Generate the JSON tool definition now."
    )

    system_prompt = _build_system_prompt(confidence_scores)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    response = client.chat(messages)
    raw_content = client.get_content(response)

    schema: dict[str, Any] | None = None
    parse_error: str | None = None

    try:
        schema = _parse_schema_json(raw_content)
        _validate_schema(schema)
    except (json.JSONDecodeError, ValueError) as e:
        parse_error = str(e)
        logger.warning("Schema parse failed, retrying: %s", parse_error)

    if schema is None:
        retry_messages = messages + [
            {"role": "assistant", "content": raw_content},
            {"role": "user", "content": RETRY_PROMPT.format(error=parse_error)},
        ]
        response2 = client.chat(retry_messages)
        raw_content2 = client.get_content(response2)
        try:
            schema = _parse_schema_json(raw_content2)
            _validate_schema(schema)
        except (json.JSONDecodeError, ValueError) as e:
            raise ValueError(f"Schema generation failed after retry: {e}") from e

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(schema, f, indent=2)
            f.write("\n")

    props = schema.get("parameters", {}).get("properties", {})
    data_fields = [k for k in props if not k.endswith("_confidence")]
    print(f"Generated schema '{schema['name']}' with {len(data_fields)} data fields:")
    for field in data_fields:
        prop = props[field]
        print(f"  - {field}: {prop.get('type')} — {prop.get('description', '')[:60]}")
    if output_path:
        print(f"\nSchema written to: {output_path}")

    return schema
