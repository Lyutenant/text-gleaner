# CLAUDE.md — textgleaner

## What this project is

A two-phase CLI tool and Python library for structured data extraction from plain-text documents using LLM tool calls.

- **Phase 1 (`generate-schema`)**: LLM analyzes sample text files + a user description to generate a JSON tool-call schema
- **Phase 2 (`extract`)**: LLM extracts structured data from text files by being forced to call the tool defined in the schema

The "forced tool call" trick: when `tool_choice` is set to require a specific tool, the LLM must populate that tool's arguments — giving deterministic, schema-validated JSON output.

**Input is always plain text.** PDF-to-text conversion, chunking, and any other pre-processing is the user's responsibility.

---

## LLM Configuration

The LLM is a **local Qwen model** served via Ollama. All calls go through Ollama's OpenAI-compatible HTTP API at `/v1/chat/completions`.

Config lives in `config.yaml` (gitignored; copy from `config.example.yaml`):

```yaml
llm:
  base_url: "http://<host>:<port>"
  model: "qwen3.5:35b-a3b-q4_K_M"
  api_key: "local"
  temperature: 0.2
  max_tokens: 32768
  timeout_seconds: 1800

extraction:
  confidence_scores: true
  max_chars: 200000
```

**`config.yaml` is only read by the CLI** (`cli.py` loads it explicitly). The Python API (`generate_schema()`, `extract()`) reads configuration from **environment variables** only (`TEXTGLEANER__LLM__BASE_URL`, `TEXTGLEANER__EXTRACTION__MAX_CHARS`, etc.), or from kwargs passed directly to the function. If you use the Python API and want to load from `config.yaml`, read it yourself and pass the values as kwargs.

**Never hardcode the base URL.** Always load from config.

### Why `/v1/chat/completions` and not `/api/chat`

Ollama exposes two APIs. The native `/api/chat` silently ignores `tool_choice`, so the model freely returns plain-text instead of a tool call. The OpenAI-compatible `/v1/chat/completions` properly enforces `tool_choice`. Always use the `/v1/` endpoint.

### Qwen3 thinking mode

Qwen3 models have an extended reasoning ("thinking") mode that consumes tokens before generating content. It is disabled by sending `"extra_body": {"think": false}` in the request payload to prevent token budget exhaustion on large documents.

### Streaming to avoid TCP timeouts

All LLM requests use `"stream": true`. Without streaming, Ollama generates the entire response server-side before sending the first byte. On remote or VPN connections (e.g. Tailscale) the TCP connection goes idle during generation and times out before any response arrives — even with a long `timeout_seconds`. With streaming, the server sends tokens as they are generated, keeping the connection alive throughout. `LLMClient.chat()` reassembles the SSE stream into the same response dict shape as a non-streaming call, so `get_content()` and `get_tool_arguments()` are unchanged.

---

## Architecture

```
textgleaner/
├── cli.py              # typer entrypoint; two commands: generate-schema, extract
├── config.py           # pydantic-settings model; env var overrides only (no YAML)
├── llm_client.py       # thin httpx wrapper; OpenAI-compatible /v1/chat/completions
├── schema_generator.py # Phase 1 logic
├── extractor.py        # Phase 2 logic
└── __init__.py         # public Python API: generate_schema(), extract()
```

---

## Public Python API

```python
from textgleaner import generate_schema, extract, Text

# Phase 1
schema = generate_schema(
    samples=["jan.txt", "feb.txt"],   # list of paths (str/Path) or Text instances
    description="...",                 # raw string OR path to a .yaml/.md file
    output="schema.json",              # optional
)

# Phase 2 — single file → flat dict
result = extract("jan.txt", schema=schema)

# Phase 2 — multiple files → {filename: dict}
results = extract(["jan.txt", "feb.txt"], schema=schema, output="results.json")

# Override size limit
result = extract("big.txt", schema=schema, max_chars=500_000)

# Sectionized extraction with Text — pass raw text slices directly
pages = open("statement.txt").read().split("\f")   # split on form-feed
result = extract(
    Text("".join(pages[4:8]), name="holdings"),
    schema=holdings_schema,
)

# Multiple Text sections → {name: dict}
results = extract(
    [Text(holdings_text, name="holdings"), Text(activity_text, name="activity")],
    schema=combined_schema,
)
```

### The `Text` class

`Text(content, name="<text>")` wraps a raw string so it can be passed anywhere a file path is accepted. The `name` is used as the dict key in multi-input results and in log messages. This is the primary mechanism for sectionized extraction (slicing a document before calling `extract()`).

---

## CLI

```bash
textgleaner generate-schema \
  --samples sample1.txt sample2.txt \
  --description description.yaml \
  --output schema.json

textgleaner extract \
  --inputs jan.txt feb.txt \
  --schema schema.json \
  --output results.json
```

---

## Key Design Decisions

### Forced tool call for JSON extraction
In Phase 2, the schema is passed as a tool definition with:
```python
"tool_choice": {"type": "function", "function": {"name": schema["name"]}}
```
The LLM is asked to call this tool, and extracted data is parsed from `tool_calls[0]["function"]["arguments"]`.

**Fallback:** Qwen3 via Ollama sometimes ignores `tool_choice` and returns the JSON in the message `content` field instead. `get_tool_arguments()` detects this and parses the content as JSON (stripping markdown code fences if present) before raising an error. The structured data is identical either way.

### Input size limit
If an input file exceeds `extraction.max_chars` (default 200,000), a `ValueError` is raised before the LLM call. Set to 0 to disable. Users are expected to split large files themselves.

### Null + confidence pattern
Every data field has a sibling `<field>_confidence` field (0–1 float) when `extraction.confidence_scores: true`:
- `1.0` = explicitly stated verbatim
- `0.7` = clearly implied
- `0.4` = inferred / uncertain
- `0.0` = not found (field will be `null`)

### Two-pass schema generation (Phase 1)

Schema generation uses two LLM calls:

1. **Pass 1 — structural analysis**: the LLM is asked to read the sample text and produce a detailed plain-text analysis (sections, data patterns, array fields, nesting, period/YTD variants). The prompt explicitly lists what to cover so the model doesn't skip sections.

2. **Pass 2 — schema design**: the analysis from Pass 1 is fed to a second prompt that generates the JSON schema. By separating "understand the document" from "design the schema", each pass is a simpler task and the resulting schema is more complete and correctly structured.

All sample text is sent to Pass 1. There is no internal sampling or page limiting — the user selects which files to use as samples.

### Invalid JSON retry
If Pass 2 returns malformed JSON, the conversation is extended with an error-correction prompt and the model retries once. If the retry also fails, a `ValueError` is raised.

---

## Error Handling

- **Empty input file**: log warning, skip (Phase 1) or raise (Phase 2)
- **Input exceeds max_chars**: raise `ValueError` with clear message before LLM call
- **Malformed tool call response**: log filename + error, re-raise
- **Invalid JSON from Phase 1**: retry once with correction prompt, then raise
- Logging uses stdlib `logging` with structured fields: `filename`, `error`

---

## Output

- `generate_schema()` returns the schema dict; writes to file if `output=` is given
- `extract()` with single input returns a flat dict
- `extract()` with multiple inputs returns `{filename: dict, ...}`
- When `output=` is given, the return value is written as JSON

---

## Known Limitations

- **Per-row detail quality degrades on long documents.** On 28-page documents (~120K chars), high-level summary fields extract reliably but per-security holdings and individual transaction rows are prone to hallucination. The model loses attention over dense tabular data far into a long context. Mitigation: split documents into sections (e.g. pass the holdings pages separately from the activity pages) before calling `extract()`. Chunking is the caller's responsibility.

---

## Out of Scope (do not build)

- PDF reading or any file format conversion
- Chunking or merging of chunked results
- Sampling logic
- Caching of extracted text
- Interactive schema interview mode
- GUI or web interface
- Any cloud LLM integration (local model only)

---

## Running the Tool

```bash
pip install -e .

# Phase 1
textgleaner generate-schema \
  --samples sample1.txt sample2.txt \
  --description description.yaml \
  --output schema.json

# Phase 2
textgleaner extract \
  --inputs statement.txt \
  --schema schema.json \
  --output result.json
```

---

## Testing

```bash
pytest tests/
```

Tests mock all LLM calls. Cover:
- Schema JSON parsing and validation
- Two-pass schema generation (analysis call + schema call, retry on bad JSON)
- Size limit enforcement
- Config defaults and env var overrides (`TEXTGLEANER__LLM__*`, `TEXTGLEANER__EXTRACTION__*`)
- `LLMClient` kwarg precedence over env vars
- Python public API: single vs multiple inputs, `Text` instances, `base_url` kwarg passthrough
