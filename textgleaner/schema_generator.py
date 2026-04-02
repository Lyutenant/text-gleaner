from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any

from .config import ExtractionConfig
from .llm_client import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pass 1 — Document structure analysis
# ---------------------------------------------------------------------------

ANALYSIS_SYSTEM_PROMPT = """\
You are a document structure analyst. Your job is to read sample document text and a description, \
then produce a precise structural analysis that will be used to design a data extraction schema.

Your analysis MUST cover:

1. **Sections** — list every distinct section in the document (e.g. "Portfolio Summary", \
"Holdings", "Activity"). For each section, note whether it appears once or repeats \
(e.g. once per account).

2. **Data patterns** — for each section, classify the data shape:
   - Key-value pairs (e.g. "Ending Value: $1,234")
   - Table / array of records (e.g. one row per security, one row per transaction)
   - Nested grouping (e.g. accounts contain sub-sections like Holdings and Activities)

3. **Repeating records** — for array-type sections, list the fields present in each row \
(e.g. a transaction row has: date, security name, symbol, quantity, price, amount).

4. **Multi-value patterns** — flag any field that appears more than once with different \
contexts, e.g.:
   - "This Period" vs "Year-to-Date" columns
   - Beginning vs Ending values
   - Per-account vs portfolio-wide totals

5. **Hierarchy** — describe the nesting structure (e.g. portfolio → accounts → \
holdings per account, activities per account).

6. **Field inventory** — for each section, enumerate every distinct data field you see, \
grouped by section. Be exhaustive — do not summarise or skip fields.

Write your analysis in plain text. Be specific and concrete — name the actual fields \
and sections as they appear in the document.
"""

ANALYSIS_USER_TEMPLATE = """\
Document description:
{description}

Sample document text:
{sample_text}

Produce your structural analysis now.
"""

# ---------------------------------------------------------------------------
# Pass 2 — Schema generation
# ---------------------------------------------------------------------------

SCHEMA_SYSTEM_PROMPT = """\
You are a JSON schema designer. Given a document description and a detailed structural \
analysis of a sample document, generate a single JSON object that is a valid \
OpenAI-compatible tool/function definition for structured data extraction.

Schema design rules:
- Return ONLY valid JSON — no markdown fences, no commentary.
- Top-level keys: "name" (snake_case identifier), "description" (one sentence), \
"parameters" (JSON Schema object with "type": "object" and "properties").
- Use nested objects for logically grouped data (e.g. portfolio_summary, account_value).
- Use arrays of objects for repeating records (e.g. holdings, activities, accounts).
- For fields that appear in both "This Period" and "Year-to-Date" columns, create \
separate sibling fields: "foo_period" and "foo_ytd".
- For optional fields use type ["string", "null"]; for required string fields use "string".
- Every property must have a "description" stating where in the document to find it.
- Array item schemas must have "type": "object" with "properties" listing every \
field in a single record.
- Be exhaustive — capture every field from the structural analysis. Do not omit \
fields to save space.
{confidence_instruction}
"""

CONFIDENCE_INSTRUCTION = """\
- For each leaf data field "foo", add a sibling field "foo_confidence" with \
"type": "number" and "description": "Confidence 0-1: 1.0=verbatim, 0.7=implied, \
0.4=inferred, 0.0=not found".
"""

SCHEMA_USER_TEMPLATE = """\
Document description:
{description}

Structural analysis of sample document:
{analysis}

Generate the JSON tool definition now.
"""

RETRY_PROMPT = """\
The previous response was not valid JSON. Error: {error}

Return ONLY a valid JSON object — no markdown, no explanation. Try again.
"""


def _build_schema_system_prompt(confidence_scores: bool) -> str:
    ci = CONFIDENCE_INSTRUCTION if confidence_scores else ""
    return SCHEMA_SYSTEM_PROMPT.format(confidence_instruction=ci)


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


def _run_analysis(client: LLMClient, description: str, sample_text: str) -> str:
    """Pass 1: ask the LLM to analyse the document structure."""
    messages = [
        {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": ANALYSIS_USER_TEMPLATE.format(
            description=description,
            sample_text=sample_text,
        )},
    ]
    logger.info("Pass 1: analysing document structure…")
    response = client.chat(messages)
    analysis = client.get_content(response)
    logger.debug("Structure analysis:\n%s", analysis[:1000])
    return analysis


def _run_schema_generation(
    client: LLMClient,
    description: str,
    analysis: str,
    confidence_scores: bool,
) -> tuple[dict[str, Any], list[dict]]:
    """Pass 2: generate the schema from the structural analysis.

    Returns (schema_dict, messages) so the retry path can extend the conversation.
    """
    system_prompt = _build_schema_system_prompt(confidence_scores)
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": SCHEMA_USER_TEMPLATE.format(
            description=description,
            analysis=analysis,
        )},
    ]
    logger.info("Pass 2: generating schema…")
    response = client.chat(messages)
    raw = client.get_content(response)

    try:
        schema = _parse_schema_json(raw)
        _validate_schema(schema)
        return schema, messages
    except (json.JSONDecodeError, ValueError) as e:
        parse_error = str(e)
        logger.warning("Schema parse failed, retrying: %s", parse_error)

    # Retry within the same conversation
    retry_messages = messages + [
        {"role": "assistant", "content": raw},
        {"role": "user", "content": RETRY_PROMPT.format(error=parse_error)},
    ]
    response2 = client.chat(retry_messages)
    raw2 = client.get_content(response2)
    try:
        schema = _parse_schema_json(raw2)
        _validate_schema(schema)
        return schema, retry_messages
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Schema generation failed after retry: {e}") from e


def generate_schema(
    samples: list[tuple[str, str]],
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
    for text, name in samples:
        text = text.strip()
        if text:
            snippets.append(f"=== {name} ===\n{text}")
        else:
            logger.warning("filename=%s error=empty_file", name)

    if not snippets:
        raise ValueError("No readable text found in any sample file.")

    sample_text = "\n\n".join(snippets)

    # Pass 1: structural analysis
    analysis = _run_analysis(client, description, sample_text)

    # Pass 2: schema generation
    schema, _ = _run_schema_generation(client, description, analysis, confidence_scores)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(schema, f, indent=2)
            f.write("\n")

    props = schema.get("parameters", {}).get("properties", {})
    data_fields = [k for k in props if not k.endswith("_confidence")]
    print(f"Generated schema '{schema['name']}' with {len(data_fields)} top-level fields:")
    for field in data_fields:
        prop = props[field]
        ftype = prop.get("type", prop.get("items", {}).get("type", ""))
        print(f"  - {field}: {ftype} — {prop.get('description', '')[:60]}")
    if output_path:
        print(f"\nSchema written to: {output_path}")

    return schema
