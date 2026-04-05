# CLAUDE.md — textgleaner

## What this project is

A two-phase CLI tool and Python library for structured data extraction from plain-text documents using LLM tool calls.

- **Phase 1 (`generate-schema`)**: LLM analyzes sample text files + a user description to generate a JSON tool-call schema
- **Phase 2 (`extract`)**: LLM extracts structured data from text files using the schema, via forced tool call or grammar-constrained structured output

**Input is always plain text.** PDF-to-text conversion, chunking, and any other pre-processing is the user's responsibility.

---

## LLM Configuration

The LLM is a **local model** served via Ollama. All calls go through Ollama's OpenAI-compatible HTTP API at `/v1/chat/completions`.

Config lives in `config.yaml` (gitignored; copy from `config.example.yaml`):

```yaml
llm:
  base_url: "http://localhost:11434"
  model: "qwen3:30b"
  api_key: "local"
  temperature: 0.2
  max_tokens: 32768
  timeout_seconds: 1800

extraction:
  confidence_scores: true
  max_chars: 200000
  extraction_method: tool_call  # tool_call | structured_output | auto
```

The Python API supports three configuration methods, in priority order (highest first):
1. **Explicit kwargs** passed directly to `generate_schema()` / `extract()` (e.g. `base_url="..."`)
2. **`Config` object** passed as `config=` — either `Config.from_yaml("config.yaml")` or `Config(base_url="...", model="...")`
3. **Environment variables** (`TEXTGLEANER__LLM__BASE_URL`, `TEXTGLEANER__EXTRACTION__MAX_CHARS`, etc.)
4. **Hardcoded defaults** in `config.py`

The CLI reads `config.yaml` via `Config.from_yaml()` and passes it as `config=`. If no `--config` flag is given, it silently looks for `config.yaml` in the current directory and falls back to env vars / defaults if not found.

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
├── extractor.py        # Phase 2 logic: _extract_one_tool_call, _extract_one_structured
├── reporter.py         # summarize(), write_csv(), write_excel(), write_summary_csv()
└── __init__.py         # public Python API: Config, generate_schema(), extract(), summarize(), Text
```

---

## Public Python API

```python
from textgleaner import Config, generate_schema, extract, Text

# --- Configuration ---

# Option 1: load from YAML
cfg = Config.from_yaml("config.yaml")

# Option 2: set directly in code
cfg = Config(base_url="http://localhost:11434", model="qwen3:30b")

# Option 3: no Config object — reads from env vars / defaults
# (just omit config= from the calls below)

# --- Phase 1: generate schema ---
schema = generate_schema(
    samples=["jan.txt", "feb.txt"],   # list of paths (str/Path) or Text instances
    description="...",                 # raw string OR path to a .yaml/.md file
    output="schema.json",              # optional
    config=cfg,
)

# --- Phase 2: extract ---

# Single file → flat dict
result = extract("jan.txt", schema=schema, config=cfg)

# Multiple files → {filename: dict}
results = extract(["jan.txt", "feb.txt"], schema=schema, output="results.json", config=cfg)

# Override a specific value (explicit kwargs take priority over config)
result = extract("big.txt", schema=schema, config=cfg, max_chars=500_000)

# Use structured output (grammar-constrained) instead of tool call
result = extract("doc.txt", schema=schema, config=cfg, extraction_method="structured_output")

# Sectionized extraction with Text — pass raw text slices directly
pages = open("statement.txt").read().split("\f")   # split on form-feed
result = extract(
    Text("".join(pages[4:8]), name="holdings"),
    schema=holdings_schema,
    config=cfg,
)

# Multiple Text sections → {name: dict}
results = extract(
    [Text(holdings_text, name="holdings"), Text(activity_text, name="activity")],
    schema=combined_schema,
    config=cfg,
)
```

### The `Config` class

`Config` holds all LLM and extraction settings. Priority order: explicit kwargs > `Config` object > env vars > defaults.

- `Config.from_yaml(path)` — load from a YAML file (raises `FileNotFoundError` if missing)
- `Config(base_url=..., model=..., ...)` — set values directly in code
- `Config()` — falls back entirely to env vars and defaults

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

# Custom config file (default: config.yaml in cwd, silently ignored if absent)
textgleaner --config /path/to/myconfig.yaml extract \
  --inputs statement.txt \
  --schema schema.json
```

---

## Key Design Decisions

### Extraction methods (Phase 2)

Two methods are available via `extraction_method` config / kwarg:

**`tool_call`** (default): The schema is passed as a tool definition with forced `tool_choice`. The LLM is instructed to call the tool, and extracted data is parsed from `tool_calls[0]["function"]["arguments"]`. Enforcement is by instruction-following — the model reads "you must call this tool" and complies. Works well with large, capable models (Qwen3, GPT-4o, etc.).

**Fallback within tool_call:** Qwen3 via Ollama sometimes ignores `tool_choice` and returns the JSON in the message `content` field instead. `get_tool_arguments()` detects this and parses the content as JSON (stripping markdown code fences if present).

**`structured_output`**: The schema is passed as `response_format: {type: json_schema, json_schema: {...}}`. Ollama hands this to llama.cpp, which converts the schema into a grammar and applies it during token sampling — the model physically cannot produce output that doesn't match the schema. More reliable on smaller or weaker models that may ignore `tool_choice`. If the model returns empty content (observed occasionally with Qwen3 + `think: false`), the call is retried once automatically.

**`auto`**: Tries `tool_call` first. Falls back to `structured_output` if the model returns unparseable output (`ValueError`, `JSONDecodeError`) or the server rejects the tools payload (`HTTP 400/422`). All other failures (timeouts, `HTTP 5xx`) are re-raised immediately since they would fail the same way on either path.

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
- **Empty content in structured_output**: retry once automatically, then raise
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
- `Config` class: direct kwargs, `from_yaml()`, missing file, partial YAML
- `config=` kwarg on public API: values passed through to `LLMClient`, explicit kwarg overrides config
- `extraction_method`: tool_call path, structured_output path, response_format payload shape, markdown fence stripping, auto routing (success, ValueError fallback, HTTP 400 fallback, HTTP 500 re-raise)
- `reporter`: summarize() null-rate/confidence, CSV output, empty list as null, confidence field exclusion, public summarize() with file output, extract() CSV output integration
- Python public API: single vs multiple inputs, `Text` instances, `base_url` kwarg passthrough

---

## To-Do / Roadmap

Items suggested but not yet implemented, roughly in priority order:

### Near-term
- [ ] **GitHub Actions CI** — run `pytest` on every push; makes the repo look maintained and catches regressions
- [ ] **PyPI publishing** — `pip install textgleaner` from PyPI; `pyproject.toml` is already set up, just needs a publish workflow
- [ ] **Examples gallery** — `examples/` directory with 2–3 real-world use cases (invoice, contract, brokerage statement) including sample description files, schemas, and a walkthrough README

### Medium-term
- [ ] **Retry on low-confidence fields** — after extraction, detect fields with confidence ≤ 0.4 and re-prompt with only those fields; one targeted follow-up call often recovers missed values
- [ ] **Schema validation / dry-run** — a `validate` command that runs extraction and reports which fields came back null or low-confidence; helps users iterate on their schema before a full batch run
- [x] **Batch extraction with summary report** — `--inputs-dir` in CLI; output format inferred from extension (`.json`/`.csv`/`.xlsx`); `summarize()` computes per-field null-rate and avg confidence; `--report` writes summary CSV
- [x] **Make `auto` mode smarter** — tries `tool_call` first; falls back to `structured_output` on `ValueError`, `JSONDecodeError`, or `HTTP 400/422`; re-raises timeouts and `HTTP 5xx`

### Longer-term
- [ ] **Schema versioning / refinement** — a `refine-schema` command that takes an existing schema + new samples and patches it, without re-running Phase 1 from scratch
- [ ] **Model profiles** — abstract Qwen3-specific workarounds (`extra_body: {think: false}`) into named model profiles so other models (Llama, Mistral, etc.) work better out of the box
- [ ] **Streaming extraction output** — progress callback / hook so callers can process partial results as each input completes, rather than waiting for the full batch
