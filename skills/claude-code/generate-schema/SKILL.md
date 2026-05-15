---
name: generate-schema
description: Generate a textgleaner-compatible JSON extraction schema from plain-text sample documents. Use when the user wants to create a schema for structured data extraction from documents like invoices, statements, or contracts.
argument-hint: [sample-file.txt ...] [what to extract]
disable-model-invocation: true
---

Arguments: $ARGUMENTS

If no arguments were provided above, ask the user for:
1. **Sample documents** — file paths to read (plain text only; PDFs must be pre-converted)
2. **Extraction description** — what to extract, e.g. "all transactions and portfolio totals"

Do not proceed until you have both.

---

## Pass 1 — Structural analysis

Read every sample file provided. Produce a detailed plain-text analysis covering each point below. Do not abbreviate — completeness here determines schema quality.

**Sections:** List every distinct section. Note whether each appears once or repeats.

**Data patterns per section:**
- Key-value pairs (e.g. `Ending Value: $1,234`)
- Table / array of records (e.g. one row per security)
- Nested grouping (e.g. accounts each containing Holdings and Activity sub-tables)

**Repeating records:** For every array-type section, list every field in a single record row (e.g. `date`, `description`, `symbol`, `quantity`, `price`, `amount`).

**Multi-value patterns:** Flag any field appearing with multiple contexts:
- Column variants: "This Period" vs "Year-to-Date"
- Temporal variants: "Beginning Value" vs "Ending Value"
- Scope variants: per-account vs portfolio-wide

**Hierarchy:** Describe the nesting structure (e.g. `portfolio → accounts → holdings[], transactions[]`).

**Field inventory:** List every distinct data field grouped by section. Be exhaustive — do not omit any field.

---

## Pass 2 — Schema design

Using the structural analysis and the extraction description, produce a single JSON object.

### Required format

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

### Rules

- `name` — snake_case, e.g. `extract_brokerage_statement`
- `description` — one sentence summary
- **Nested objects** for logically grouped key-value data
- **Arrays of objects** for repeating records; `items` must define every field in one record
- **Multi-value fields**: use separate siblings (`income_this_period`, `income_ytd`), never a single ambiguous field
- Optional fields: `"type": ["string", "null"]`; required strings: `"type": "string"`
- Every property needs a `"description"` stating where in the document to find it
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

---

## Output

If the user specified an output path, write the schema JSON there. Then print:

```
Generated schema 'extract_...' with N top-level fields:
  - field_name: type — description (first 60 chars)
  ...
```

The schema is directly usable with `textgleaner extract` or the `extract()` Python API.
