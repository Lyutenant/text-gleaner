from __future__ import annotations
import json
import logging
import random
from pathlib import Path
from typing import Any

import yaml

from .config import get_config
from .llm_client import LLMClient
from .pdf_reader import extract_text_from_pdf, chunk_pages

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a data extraction schema designer. Given sample PDF text and a document description, \
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


def _sample_pdfs(pdfs_dir: Path, sample_ratio: float, sample_dir: Path | None) -> list[Path]:
    if sample_dir is not None:
        return sorted(sample_dir.glob("*.pdf"))
    all_pdfs = sorted(pdfs_dir.glob("*.pdf"))
    if sample_ratio >= 1.0 or not all_pdfs:
        return all_pdfs
    k = max(1, int(len(all_pdfs) * sample_ratio))
    return random.sample(all_pdfs, k)


def _extract_snippet(path: Path, max_pages: int = 5) -> str:
    pages = extract_text_from_pdf(path)
    if not pages:
        return ""
    snippet_pages = pages[:max_pages]
    return "\n\n".join(p for p in snippet_pages if p.strip())


def _parse_schema_json(text: str) -> dict:
    text = text.strip()
    # Strip markdown fences if present
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)


def _validate_schema(schema: dict) -> None:
    for key in ("name", "description", "parameters"):
        if key not in schema:
            raise ValueError(f"Schema missing required key: '{key}'")
    params = schema["parameters"]
    if "properties" not in params:
        raise ValueError("Schema 'parameters' missing 'properties'")


def generate_schema(
    pdfs_dir: Path,
    description_file: Path,
    output_file: Path,
    sample_ratio: float = 1.0,
    sample_dir: Path | None = None,
) -> dict:
    cfg = get_config()
    client = LLMClient()

    # Load description
    with description_file.open() as f:
        if description_file.suffix in (".yaml", ".yml"):
            desc_content = yaml.safe_load(f)
            desc_text = yaml.dump(desc_content, default_flow_style=False)
        else:
            desc_text = f.read()

    # Collect sample PDFs
    sample_pdfs = _sample_pdfs(pdfs_dir, sample_ratio, sample_dir)
    if not sample_pdfs:
        raise ValueError(f"No PDFs found in {pdfs_dir}")

    logger.info("Using %d sample PDFs for schema generation", len(sample_pdfs))

    # Build sample text snippets
    snippets: list[str] = []
    for pdf_path in sample_pdfs:
        snippet = _extract_snippet(pdf_path)
        if snippet:
            snippets.append(f"=== {pdf_path.name} ===\n{snippet}")
        else:
            logger.warning("filename=%s error=no_extractable_text", pdf_path.name)

    sample_text = "\n\n".join(snippets) if snippets else "(no sample text extracted)"

    user_message = (
        f"Document description:\n{desc_text}\n\n"
        f"Sample PDF text snippets:\n{sample_text}\n\n"
        "Generate the JSON tool definition now."
    )

    system_prompt = _build_system_prompt(cfg.extraction.confidence_scores)
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
        # Retry once
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

    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        json.dump(schema, f, indent=2)
        f.write("\n")

    # Print summary
    props = schema.get("parameters", {}).get("properties", {})
    data_fields = [k for k in props if not k.endswith("_confidence")]
    print(f"Generated schema '{schema['name']}' with {len(data_fields)} data fields:")
    for field in data_fields:
        prop = props[field]
        print(f"  - {field}: {prop.get('type')} — {prop.get('description', '')[:60]}")
    print(f"\nSchema written to: {output_file}")

    return schema
