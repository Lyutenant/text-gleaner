---
name: generate_schema
description: Generate a structured data extraction schema from plain-text sample documents and a user-provided description of what to extract. Produces a textgleaner-compatible JSON schema without invoking any external tool or library.
---

## When to use this skill

Use this skill when a user wants to create a schema for structured data extraction from plain-text documents (invoices, statements, contracts, reports, etc.) and has at least one sample document available.

## Inputs required

Before starting, confirm you have both:

1. **Sample documents** — one or more plain-text files. Ask the user for file paths or have them paste the content directly. PDFs must be pre-converted to text.
2. **Extraction description** — what the user wants to extract. Accept a brief phrase ("extract all transactions and totals") or a detailed YAML description.

If either is missing, ask for it before proceeding.

## Step 1 — Read the samples

Read every sample file the user provided. If content was pasted directly, treat it as the sample text. Label each sample by filename or a short name.

## Step 2 — Structural analysis

Carefully analyze all sample text. Write a detailed plain-text analysis covering each of the following. Do not skip or abbreviate — completeness here determines schema quality.

**Sections:** List every distinct section (e.g. "Portfolio Summary", "Holdings", "Activity"). Note whether each appears once or repeats.

**Data patterns:** For each section, classify the data shape:
- Key-value pairs (e.g. `Ending Value: $1,234`)
- Table / array of records (one row per security, one row per transaction)
- Nested grouping (e.g. accounts each containing Holdings and Activity sub-tables)

**Repeating records:** For every array-type section, list every field in a single record row (e.g. a transaction row: `date`, `description`, `symbol`, `quantity`, `price`, `amount`).

**Multi-value patterns:** Flag any field that appears multiple times with different contexts:
- Column variants: "This Period" vs "Year-to-Date"
- Temporal variants: "Beginning Value" vs "Ending Value"
- Scope variants: per-account vs portfolio-wide

**Hierarchy:** Describe the nesting structure (e.g. `portfolio → accounts → holdings[], transactions[]`).

**Field inventory:** List every distinct data field grouped by section. Be exhaustive — do not omit any field.

## Step 3 — Generate the schema

Using the structural analysis and the user's extraction description, produce a single JSON object following the rules below.

### Schema format

```json
{
  "name": "extract_something",
  "description": "One sentence describing what this schema extracts.",
  "parameters": {
    "type": "object",
    "properties": {
      "field_name": {
        "type": "string",
        "description": "Where in the document this value comes from."
      }
    },
    "required": []
  }
}
```

### Design rules

- `name` — snake_case identifier, e.g. `extract_brokerage_statement`
- `description` — one sentence summary
- Use **nested objects** for logically grouped key-value data
- Use **arrays of objects** for repeating records; array `items` must have `"type": "object"` with `"properties"` for every field in one record
- **Multi-value fields**: create separate siblings — `income_this_period` and `income_ytd` — not a single ambiguous field
- Optional fields: `"type": ["string", "null"]`; required string fields: `"type": "string"`
- Every property must have a `"description"` stating where in the document to find it
- Be exhaustive — do not omit fields to save space
- Output only valid JSON — no markdown fences, no commentary

### Confidence fields

Unless the user opts out, add a `_confidence` sibling immediately after every leaf scalar field:

```json
"amount": {
  "type": ["number", "null"],
  "description": "Transaction amount in dollars."
},
"amount_confidence": {
  "type": "number",
  "description": "Confidence 0-1: 1.0=verbatim, 0.7=implied, 0.4=inferred, 0.0=not found"
}
```

Do not add confidence fields for array container fields or for other `_confidence` fields.

## Step 4 — Output

Write the schema JSON to a file if the user specified a path. Then print a summary:

```
Generated schema 'extract_...' with N top-level fields:
  - field_name: type — description (first 60 chars)
  ...
```

## Notes

- The output schema is directly usable with `textgleaner extract` or the `extract()` Python API.
- If samples are too short or unrepresentative, say so and ask for better samples before proceeding.
- If the extraction description is ambiguous, ask one focused clarifying question before starting Step 2.
