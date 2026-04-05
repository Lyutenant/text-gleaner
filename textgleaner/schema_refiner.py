"""Schema refinement: update an existing schema from new sample documents."""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any

from .config import ExtractionConfig
from .llm_client import LLMClient
from .schema_generator import _parse_schema_json, _validate_schema, RETRY_PROMPT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pass 1 — Gap analysis
# ---------------------------------------------------------------------------

GAP_ANALYSIS_SYSTEM_PROMPT = """\
You are a schema gap analyst. You will be given an existing JSON extraction schema and \
one or more new sample documents. Your job is to compare the samples against the schema \
and produce a precise gap analysis.

Your analysis MUST cover:

1. **Missing fields** — fields that appear in the samples but are NOT in the schema. \
For each, state the proposed field name, where it appears in the document, its type \
(string / number / array / object), and one or two sample values.

2. **Type mismatches** — fields whose types in the schema do not match what actually \
appears in the samples (e.g. schema says "string" but values are always numeric, or a \
field is singular but always repeats as a list).

3. **Dead fields** — fields in the schema that are never populated in these samples. \
Note them, but do NOT recommend removing a field unless it also contradicts the \
document structure (it may simply be absent from these particular samples).

4. **Structural issues** — e.g. a field that should be an array of objects (repeating \
records) but is typed as a single string, or a flat field that belongs inside a nested object.

5. **Description improvements** — schema field descriptions that are inaccurate, \
incomplete, or misleading given what you see in the samples.

Be specific and concrete — reference actual text from the samples.
If the schema already covers everything in the new samples, say so explicitly.
"""

GAP_ANALYSIS_USER_TEMPLATE = """\
Existing schema:
{schema_json}

New sample documents:
{sample_text}

Produce your gap analysis now.
"""

# ---------------------------------------------------------------------------
# Pass 2 — Schema refinement
# ---------------------------------------------------------------------------

REFINEMENT_SYSTEM_PROMPT = """\
You are a JSON schema refinement assistant. Given an existing extraction schema and a \
gap analysis describing what needs to change, produce the COMPLETE updated schema JSON.

Rules:
- Return ONLY valid JSON — no markdown fences, no commentary.
- Preserve ALL existing fields unless the gap analysis explicitly recommends removal.
- Add all new fields identified in the gap analysis.
- Fix type mismatches identified in the gap analysis.
- Top-level structure: {{"name": "...", "description": "...", "parameters": {{"type": "object", "properties": {{...}}}}}}
- Use ["type", "null"] for optional fields.
- Arrays of objects: {{"type": "array", "items": {{"type": "object", "properties": {{...}}}}}}
- Every property must have a "description" stating where in the document to find it.
- Be exhaustive — do not drop existing fields silently.
{confidence_instruction}
"""

CONFIDENCE_INSTRUCTION = """\
- The existing schema uses confidence score fields. For every new leaf data field \
"foo" you add, also add a sibling "foo_confidence" with \
"type": "number" and "description": "Confidence 0-1: 1.0=verbatim, 0.7=implied, \
0.4=inferred, 0.0=not found".
"""

REFINEMENT_USER_TEMPLATE = """\
Existing schema:
{schema_json}

Gap analysis:
{analysis}

Produce the complete updated schema JSON now.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_confidence_scores(schema: dict) -> bool:
    """Return True if the schema already uses _confidence sibling fields."""
    props = schema.get("parameters", {}).get("properties", {})
    return any(k.endswith("_confidence") for k in props)


def _build_refinement_system_prompt(confidence_scores: bool) -> str:
    ci = CONFIDENCE_INSTRUCTION if confidence_scores else ""
    return REFINEMENT_SYSTEM_PROMPT.format(confidence_instruction=ci)


def _run_gap_analysis(
    client: LLMClient,
    schema: dict,
    sample_text: str,
) -> str:
    """Pass 1: ask the LLM to compare new samples against the existing schema."""
    messages = [
        {"role": "system", "content": GAP_ANALYSIS_SYSTEM_PROMPT},
        {"role": "user", "content": GAP_ANALYSIS_USER_TEMPLATE.format(
            schema_json=json.dumps(schema, indent=2),
            sample_text=sample_text,
        )},
    ]
    logger.info("Pass 1: running gap analysis…")
    response = client.chat(messages)
    analysis = client.get_content(response)
    logger.debug("Gap analysis:\n%s", analysis[:1000])
    return analysis


def _run_schema_refinement(
    client: LLMClient,
    schema: dict,
    analysis: str,
    confidence_scores: bool,
) -> dict[str, Any]:
    """Pass 2: produce the updated schema from the existing one + gap analysis.

    Retries once within the same conversation if the response is not valid JSON.
    """
    system_prompt = _build_refinement_system_prompt(confidence_scores)
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": REFINEMENT_USER_TEMPLATE.format(
            schema_json=json.dumps(schema, indent=2),
            analysis=analysis,
        )},
    ]
    logger.info("Pass 2: refining schema…")
    response = client.chat(messages)
    raw = client.get_content(response)

    try:
        updated = _parse_schema_json(raw)
        _validate_schema(updated)
        return updated
    except (json.JSONDecodeError, ValueError) as e:
        parse_error = str(e)
        logger.warning("Schema parse failed, retrying: %s", parse_error)

    retry_messages = messages + [
        {"role": "assistant", "content": raw},
        {"role": "user", "content": RETRY_PROMPT.format(error=parse_error)},
    ]
    response2 = client.chat(retry_messages)
    raw2 = client.get_content(response2)
    try:
        updated = _parse_schema_json(raw2)
        _validate_schema(updated)
        return updated
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Schema refinement failed after retry: {e}") from e


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def refine_schema(
    schema: dict,
    samples: list[tuple[str, str]],
    output_path: Path | None,
    *,
    confidence_scores: bool | None = None,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: int | None = None,
    model_profile: str | None = None,
) -> dict:
    # Auto-detect confidence_scores from the existing schema if not specified.
    if confidence_scores is None:
        confidence_scores = _detect_confidence_scores(schema)

    client = LLMClient(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        model_profile=model_profile,
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

    # Pass 1: gap analysis
    analysis = _run_gap_analysis(client, schema, sample_text)

    # Pass 2: schema refinement
    updated = _run_schema_refinement(client, schema, analysis, confidence_scores)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(updated, f, indent=2)
            f.write("\n")

    # Summary of changes
    old_props = schema.get("parameters", {}).get("properties", {})
    new_props = updated.get("parameters", {}).get("properties", {})
    old_fields = {k for k in old_props if not k.endswith("_confidence")}
    new_fields = {k for k in new_props if not k.endswith("_confidence")}
    added = new_fields - old_fields
    removed = old_fields - new_fields

    print(f"Refined schema '{updated['name']}': "
          f"{len(new_fields)} top-level fields "
          f"(+{len(added)} added, -{len(removed)} removed)")
    if added:
        for f in sorted(added):
            print(f"  + {f}")
    if removed:
        for f in sorted(removed):
            print(f"  - {f}")
    if not added and not removed:
        print("  (no top-level fields changed — descriptions or types may have been updated)")
    if output_path:
        print(f"\nSchema written to: {output_path}")

    return updated
