# textgleaner

Extract structured data from plain-text documents using a local LLM.

textgleaner uses a two-phase approach:

1. **Generate schema** — the LLM analyzes sample documents and your description to produce a JSON extraction schema
2. **Extract** — the LLM is forced to call the schema as a tool, returning deterministic, schema-validated JSON

All inference runs locally via [Ollama](https://ollama.com). No data leaves your machine.

---

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com) running locally (or on a remote host)
- A model that supports tool calls (e.g. `qwen3:30b`, `llama3.1:8b`)

---

## Installation

```bash
pip install textgleaner
```

Or from source:

```bash
git clone https://github.com/Lyutenant/text-gleaner
cd text-gleaner
pip install -e .
```

---

## Configuration

Copy the example config and edit it:

```bash
cp config.example.yaml config.yaml
```

```yaml
llm:
  base_url: "http://localhost:11434"   # Ollama default
  model: "qwen3:30b"
  api_key: "local"
  temperature: 0.2
  max_tokens: 32768
  timeout_seconds: 1800

extraction:
  confidence_scores: true
  max_chars: 200000
```

You can also configure via environment variables:

```bash
export TEXTGLEANER__LLM__BASE_URL="http://localhost:11434"
export TEXTGLEANER__LLM__MODEL="qwen3:30b"
```

---

## CLI

```bash
# Phase 1: generate a schema from sample documents
textgleaner generate-schema \
  --samples sample1.txt sample2.txt \
  --description description.yaml \
  --output schema.json

# Phase 2: extract structured data
textgleaner extract \
  --inputs statement.txt \
  --schema schema.json \
  --output result.json

# Use a custom config file
textgleaner --config myconfig.yaml extract --inputs doc.txt --schema schema.json
```

---

## Python API

### Quick start

```python
from textgleaner import Config, generate_schema, extract, Text

# Load config from YAML
cfg = Config.from_yaml("config.yaml")

# Or set values directly
cfg = Config(base_url="http://localhost:11434", model="qwen3:30b")

# Phase 1: generate a schema
schema = generate_schema(
    samples=["jan.txt", "feb.txt"],
    description="Monthly brokerage statement with holdings and transactions.",
    output="schema.json",
    config=cfg,
)

# Phase 2: extract from a single file
result = extract("statement.txt", schema=schema, config=cfg)

# Phase 2: extract from multiple files → {filename: dict}
results = extract(["jan.txt", "feb.txt"], schema=schema, output="results.json", config=cfg)
```

### Sectionized extraction with `Text`

Use `Text` to pass raw text slices directly — useful when you want to split a document before extracting:

```python
from textgleaner import Config, extract, Text

cfg = Config.from_yaml("config.yaml")

# Split a document on form-feed page breaks
pages = open("statement.txt").read().split("\f")

# Extract from a specific page range
result = extract(
    Text("".join(pages[4:8]), name="holdings"),
    schema=holdings_schema,
    config=cfg,
)

# Extract from multiple sections → {name: dict}
results = extract(
    [
        Text(holdings_text, name="holdings"),
        Text(activity_text, name="activity"),
    ],
    schema=schema,
    config=cfg,
)
```

### Confidence scores

When `confidence_scores: true`, every extracted field has a sibling `<field>_confidence` (0–1):

| Score | Meaning |
|-------|---------|
| 1.0 | Value stated verbatim |
| 0.7 | Clearly implied |
| 0.4 | Inferred / uncertain |
| 0.0 | Not found (field is `null`) |

---

## How it works

### Forced tool call

In Phase 2, the schema is registered as an LLM tool and `tool_choice` is set to require it. The LLM must populate the tool's arguments — giving deterministic, schema-validated JSON output instead of free-form text.

### Two-pass schema generation

Phase 1 uses two LLM calls:

1. **Structural analysis** — the LLM reads the sample text and produces a detailed plain-text analysis of sections, fields, data shapes, and nesting
2. **Schema design** — a second call turns the analysis into a JSON tool definition

Separating "understand the document" from "design the schema" produces more complete and correctly structured schemas.

### Streaming to prevent timeouts

All requests use HTTP streaming (`"stream": true`). Without streaming, Ollama generates the entire response server-side before sending a single byte — causing TCP timeouts on slow or remote connections before any data arrives. Streaming keeps the connection alive throughout generation.

---

## Input format

**Input is always plain text.** PDF conversion, OCR, and any other pre-processing is your responsibility. Tools like `pdftotext` (poppler) work well for PDFs with selectable text.

---

## Known limitations

- **Per-row detail degrades on long documents.** For dense tabular data (e.g. transaction histories), extract page-by-page or section-by-section rather than feeding the entire document at once. The model's attention weakens over long contexts.
- **Local models only.** No cloud LLM integration is planned.

---

## Development

```bash
pip install -e .
pytest tests/
```

---

## License

MIT
