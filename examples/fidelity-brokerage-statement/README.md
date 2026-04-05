# Example: Fidelity Brokerage Statement — Activity Extraction

This example extracts all transaction records from the **Activity section** of a
Fidelity monthly brokerage statement using textgleaner's two-phase pipeline.

## What it extracts

The Activity section of a Fidelity statement contains several subsections per account:

- Securities Bought & Sold
- Trades Pending Settlement
- Dividends, Interest & Other Income
- Other Activity (mergers, corporate actions)
- Daily Additions and Subtractions
- Debit Card Activity
- Checking Activity
- Fees and Charges

Each page is extracted independently (one LLM call per page) to keep the context
window small and improve per-row accuracy.

---

## Getting the sample document

Fidelity publishes a sample statement here:

> **https://www.fidelity.com/bin-public/060_www_fidelity_com/documents/sample-new-fidelity-acnt-stmt.pdf**

Convert it to plain text using `pdftotext` (part of the poppler package):

```bash
# macOS
brew install poppler
pdftotext -layout sample-new-fidelity-acnt-stmt.pdf statement.txt

# Ubuntu / Debian
sudo apt install poppler-utils
pdftotext -layout sample-new-fidelity-acnt-stmt.pdf statement.txt
```

Place the resulting `statement.txt` in this directory.

---

## Setup

```bash
# From the repo root
pip install -e .
cp config.example.yaml config.yaml
# Edit config.yaml with your Ollama server URL and model name
```

---

## Running

### Phase 1 — Generate the schema (optional)

A pre-generated schema (`activity_schema.json`) is included so you can skip
directly to extraction. Run this only if you want to regenerate it:

```bash
python generate_schema.py
```

### Phase 2 — Extract transactions

```bash
python extract_transactions.py
```

Results are written to `transactions_result.json` as:

```json
{
  "page_11": { "account_header": {...}, "securities_bought_sold": [...], ... },
  "page_12": { "securities_bought_sold": [...], ... },
  ...
}
```

### Switching extraction method

Open `extract_transactions.py` and change the `EXTRACTION_METHOD` variable at the top:

```python
EXTRACTION_METHOD = "tool_call"          # default — forced tool call
EXTRACTION_METHOD = "structured_output"  # grammar-constrained — better for smaller models
```

See the [textgleaner README](../../README.md) for a full explanation of the difference.

---

## Files

| File | Description |
|------|-------------|
| `description.yaml` | Document description passed to Phase 1 |
| `activity_schema.json` | Pre-generated extraction schema (34 top-level fields) |
| `generate_schema.py` | Phase 1 script — regenerates the schema from pages 11–17 |
| `extract_transactions.py` | Phase 2 script — extracts transactions page by page |
