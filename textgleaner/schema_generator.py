"""Phase 1 schema generation for textgleaner.

Generates a JSON tool-call schema from sample documents using a two-pass
LLM strategy:

**Pass 1 — Structural analysis** (:func:`_run_analysis`)
    The LLM reads all sample documents and produces a detailed plain-text
    analysis covering sections, data patterns, repeating records, multi-value
    fields (e.g. "This Period" vs "Year-to-Date"), and field inventory.
    Separating understanding from schema design keeps each pass simpler and
    produces more complete results.

**Pass 2 — Schema design** (:func:`_run_schema_generation`)
    The analysis from Pass 1 is fed to a second prompt that generates the
    JSON schema.  The schema follows the OpenAI tool-definition format::

        {
            "name": "extract_something",
            "description": "One sentence summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "field_name": {"type": "string", "description": "..."},
                    "field_name_confidence": {"type": "number", "description": "..."},
                    ...
                }
            }
        }

**Invalid JSON retry**
    If Pass 2 returns malformed JSON, the conversation is extended with an
    error-correction message and the model retries once.  If the retry also
    fails a :exc:`ValueError` is raised.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any

from .config import ExtractionConfig
from .llm_client import LLMClient, make_client

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

# Appended to SCHEMA_SYSTEM_PROMPT when confidence scores are enabled.
# Each data field gets a sibling "<field>_confidence" number field.
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

# Appended to the conversation when Pass 2 returns malformed JSON.
# Extends the same conversation rather than starting fresh so the model
# can see its own bad output and correct it.
RETRY_PROMPT = """\
The previous response was not valid JSON. Error: {error}

Return ONLY a valid JSON object — no markdown, no explanation. Try again.
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_schema_system_prompt(confidence_scores: bool) -> str:
    """Build the Pass 2 system prompt, optionally including confidence instructions.

    Args:
        confidence_scores: If ``True``, the confidence score instruction is
            appended so the model adds ``<field>_confidence`` siblings.

    Returns:
        The complete system prompt string for Pass 2.
    """
    ci = CONFIDENCE_INSTRUCTION if confidence_scores else ""
    return SCHEMA_SYSTEM_PROMPT.format(confidence_instruction=ci)


def _parse_schema_json(text: str) -> dict:
    """Parse *text* as JSON, stripping markdown code fences if present.

    Some models wrap their JSON output in triple-backtick fences despite being
    instructed not to.  This function handles both fenced and bare JSON.

    Args:
        text: Raw text returned by the LLM.

    Returns:
        Parsed dict.

    Raises:
        json.JSONDecodeError: If the text (after fence stripping) is not
            valid JSON.
    """
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Remove opening fence (e.g. "```json") and closing fence ("```").
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)


def _validate_schema(schema: dict) -> None:
    """Check that *schema* has the required top-level structure.

    A valid schema must have ``"name"``, ``"description"``, and
    ``"parameters"`` keys at the top level, and ``"properties"`` inside
    ``"parameters"``.

    Args:
        schema: The parsed schema dict to validate.

    Raises:
        ValueError: If any required key is missing.
    """
    for key in ("name", "description", "parameters"):
        if key not in schema:
            raise ValueError(f"Schema missing required key: '{key}'")
    if "properties" not in schema["parameters"]:
        raise ValueError("Schema 'parameters' missing 'properties'")


def _run_analysis(client: LLMClient, description: str, sample_text: str) -> str:
    """Pass 1: ask the LLM to analyse the document structure.

    Sends all sample documents concatenated as a single text block along with
    the user-supplied description.  The model returns a plain-text analysis
    covering sections, data patterns, arrays, multi-value fields, and a
    complete field inventory.  This analysis drives Pass 2.

    Args:
        client: An LLM client instance.
        description: The user's description of the document type and what to
            extract (plain text or YAML-formatted).
        sample_text: All sample documents concatenated, each prefixed with
            ``=== name ===``.

    Returns:
        The model's plain-text structural analysis.
    """
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
    """Pass 2: generate the JSON schema from the structural analysis.

    Sends the description and Pass 1 analysis to the model and asks it to
    produce the complete schema JSON.  If the response is not valid JSON, the
    conversation is extended with an error-correction message and the model
    retries once.

    Args:
        client: An LLM client instance.
        description: The user's document description (same as passed to Pass 1).
        analysis: The plain-text analysis produced by :func:`_run_analysis`.
        confidence_scores: Whether to include ``<field>_confidence`` sibling
            fields in the generated schema.

    Returns:
        A ``(schema_dict, messages)`` tuple.  ``messages`` is the final
        conversation history (including the retry turn if one was needed),
        which callers may inspect for debugging.

    Raises:
        ValueError: If the model returns invalid JSON on both the initial
            attempt and the retry.
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

    # Retry by extending the same conversation.  The model can see its own
    # bad output and the specific error, which helps it self-correct.
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


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_schema(
    samples: list[tuple[str, str]],
    description: str,
    output_path: Path | None,
    *,
    confidence_scores: bool | None = None,
    provider: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: int | None = None,
    model_profile: str | None = None,
) -> dict:
    """Generate an extraction schema from sample documents using two LLM passes.

    This is the internal entry point called by the public
    :func:`textgleaner.generate_schema` function after it has loaded files and
    merged configuration.  It is not part of the public API.

    The function concatenates all sample texts, runs the two-pass analysis
    and schema generation, writes the schema to *output_path* if provided,
    prints a field summary, and returns the schema dict.

    Args:
        samples: List of ``(text, name)`` tuples — one per sample document.
            Empty samples are logged and skipped.
        description: The user's description of the document type and what
            fields to extract (plain text or YAML-formatted string).
        output_path: Optional path to write the schema JSON.  Parent
            directories are created if they do not exist.
        confidence_scores: Whether to include ``<field>_confidence`` siblings.
            Falls back to :class:`~config.ExtractionConfig` default (``True``).
        provider: LLM backend (``"ollama"`` or ``"claude"``).
        base_url, model, api_key, temperature, max_tokens, timeout,
        model_profile: Passed to :func:`~llm_client.make_client`.

    Returns:
        The generated schema dict.

    Raises:
        ValueError: If no readable sample text was provided, or if schema
            generation fails after the JSON-correction retry.
    """
    if confidence_scores is None:
        confidence_scores = ExtractionConfig().confidence_scores

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

    # Build the combined sample text block.  Each sample is labelled with its
    # name so the model can refer to specific examples by name if needed.
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

    # Pass 2: schema generation (with optional JSON retry)
    schema, _ = _run_schema_generation(client, description, analysis, confidence_scores)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(schema, f, indent=2)
            f.write("\n")

    # Print a human-readable summary of what was generated.
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
