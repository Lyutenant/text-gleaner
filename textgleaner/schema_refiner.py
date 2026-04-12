"""Schema refinement: update an existing schema from new sample documents.

This module implements a two-pass schema refinement workflow that extends an
existing extraction schema without discarding what was already designed:

**Pass 1 — Gap analysis** (:func:`_run_gap_analysis`)
    The LLM compares new sample documents against the existing schema and
    produces a structured plain-text analysis identifying:

    - Missing fields (present in new samples but absent from the schema)
    - Type mismatches (wrong JSON type for a field's actual values)
    - Dead fields (in the schema but never populated by these samples)
    - Structural issues (e.g. a repeating record typed as a single string)
    - Description improvements

**Pass 2 — Schema update** (:func:`_run_schema_refinement`)
    The gap analysis is fed to a second prompt that generates the complete
    updated schema JSON.  All existing fields are preserved unless the gap
    analysis explicitly recommends removal.  New fields are added, types are
    corrected, and descriptions are improved as directed.

**Invalid JSON retry**
    If Pass 2 returns malformed JSON, the conversation is extended with an
    error-correction message and the model retries once.  If the retry also
    fails a :exc:`ValueError` is raised.

**Confidence score detection**
    If the existing schema uses ``<field>_confidence`` sibling fields, the
    refinement instruction automatically adds confidence siblings for any new
    fields added by the update.
"""
from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any

from .config import ExtractionConfig
from .llm_client import LLMClient, make_client
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

# Appended to REFINEMENT_SYSTEM_PROMPT when the existing schema uses confidence fields.
# Ensures new fields get confidence siblings to match the existing schema style.
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
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_confidence_scores(schema: dict) -> bool:
    """Return ``True`` if the existing schema uses ``_confidence`` sibling fields.

    This is used to auto-detect whether confidence scoring is enabled so the
    refinement prompt can instruct the model to add confidence fields for any
    new properties it introduces.

    Args:
        schema: The existing schema dict (top-level properties are inspected).

    Returns:
        ``True`` if at least one top-level property name ends with
        ``"_confidence"``.
    """
    props = schema.get("parameters", {}).get("properties", {})
    return any(k.endswith("_confidence") for k in props)


def _build_refinement_system_prompt(confidence_scores: bool) -> str:
    """Build the Pass 2 refinement system prompt.

    Args:
        confidence_scores: If ``True``, the confidence instruction is appended
            so the model adds ``<field>_confidence`` siblings for new fields.

    Returns:
        The complete system prompt string for Pass 2.
    """
    ci = CONFIDENCE_INSTRUCTION if confidence_scores else ""
    return REFINEMENT_SYSTEM_PROMPT.format(confidence_instruction=ci)


def _run_gap_analysis(
    client: LLMClient,
    schema: dict,
    sample_text: str,
) -> str:
    """Pass 1: ask the LLM to compare new samples against the existing schema.

    The model sees both the full existing schema JSON and the concatenated
    sample documents, and produces a structured plain-text gap analysis.

    Args:
        client: An LLM client instance.
        schema: The existing schema dict (serialized to JSON for the prompt).
        sample_text: All new sample documents concatenated, each prefixed with
            ``=== name ===``.

    Returns:
        The model's plain-text gap analysis.
    """
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
    """Pass 2: produce the updated schema from the existing one and the gap analysis.

    The model is given both the full existing schema JSON and the gap analysis
    from Pass 1, and is instructed to return the complete updated schema.
    Existing fields are preserved; new fields and fixes from the gap analysis
    are applied.

    If the response is not valid JSON, the conversation is extended with an
    error-correction message and the model retries once.

    Args:
        client: An LLM client instance.
        schema: The existing schema dict (shown to the model as context).
        analysis: The plain-text gap analysis from :func:`_run_gap_analysis`.
        confidence_scores: Whether to instruct the model to add confidence
            siblings for any new fields.

    Returns:
        The updated schema dict.

    Raises:
        ValueError: If the model returns invalid JSON on both the initial
            attempt and the retry.
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

    # Retry by extending the conversation so the model sees its own bad output.
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
    provider: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: int | None = None,
    model_profile: str | None = None,
) -> dict:
    """Update an existing schema from new sample documents using two LLM passes.

    This is the internal entry point called by the public
    :func:`textgleaner.refine_schema` function after it has loaded files and
    merged configuration.  It is not part of the public API.

    Confidence score detection is automatic: if the existing schema contains
    ``<field>_confidence`` fields, new fields added by the refinement will
    also receive confidence siblings.  Pass ``confidence_scores=False`` to
    override this.

    Args:
        schema: The existing schema dict to refine.
        samples: List of ``(text, name)`` tuples — the new sample documents.
            Empty samples are logged and skipped.
        output_path: Optional path to write the updated schema JSON.  Parent
            directories are created if they do not exist.
        confidence_scores: Whether new fields should get confidence siblings.
            ``None`` (default) auto-detects from the existing schema.
        provider: LLM backend (``"ollama"`` or ``"claude"``).
        base_url, model, api_key, temperature, max_tokens, timeout,
        model_profile: Passed to :func:`~llm_client.make_client`.

    Returns:
        The updated schema dict.

    Raises:
        ValueError: If no readable sample text was provided, or if schema
            refinement fails after the JSON-correction retry.
    """
    # Auto-detect confidence_scores from the existing schema if not specified.
    if confidence_scores is None:
        confidence_scores = _detect_confidence_scores(schema)

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

    # Build the combined sample text block.
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

    # Pass 2: schema refinement (with optional JSON retry)
    updated = _run_schema_refinement(client, schema, analysis, confidence_scores)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            json.dump(updated, f, indent=2)
            f.write("\n")

    # Print a human-readable summary of what changed.
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
