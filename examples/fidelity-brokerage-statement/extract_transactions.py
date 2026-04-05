"""
Phase 2 — Extract transaction data from all Activity pages of a Fidelity statement.

Activity pages are identified by their first meaningful content line starting with
"Activity" (covers both "Activity" opener pages and "Activity (continued)" pages).
Each page is extracted independently — one LLM call per page — keeping the context
window small and focused.

Change EXTRACTION_METHOD below to switch between:
  "tool_call"         — forced tool/function call (default; best for large models)
  "structured_output" — grammar-constrained JSON via response_format (better for
                        smaller models that may ignore tool_choice)

Usage:
    python extract_transactions.py

Reads:   activity_schema.json  (run generate_schema.py first, or use the pre-generated one)
Writes:  transactions_result.json

Requirements:
    - config.yaml in the repo root (copy from config.example.yaml)
    - statement.txt in this directory (see README for how to get it)
"""
import json
from pathlib import Path

from textgleaner import Config, Text, extract

# --- configuration -----------------------------------------------------------

EXTRACTION_METHOD = "tool_call"   # "tool_call" | "structured_output"

# -----------------------------------------------------------------------------

HERE = Path(__file__).parent
STATEMENT = HERE / "statement.txt"
SCHEMA_FILE = HERE / "activity_schema.json"
OUTPUT_FILE = HERE / "transactions_result.json"
CONFIG_FILE = HERE.parent.parent / "config.yaml"

# Lines that are part of the boilerplate page header — skip when detecting the
# section heading so we look at the actual first content line of each page.
_HEADER_PREFIXES = (
    "*** SAMPLE",
    "For informational",
    "INVESTMENT REPORT",
    "January ", "February ", "March ", "April ",
    "May ", "June ", "July ", "August ", "September ",
    "October ", "November ", "December ",
)


def _first_content_line(page_text: str) -> str:
    for line in page_text.strip().splitlines():
        stripped = line.strip()
        if stripped and not any(stripped.startswith(p) for p in _HEADER_PREFIXES):
            return stripped
    return ""


def is_activity_page(page_text: str) -> bool:
    return _first_content_line(page_text).startswith("Activity")


def main() -> None:
    if not STATEMENT.exists():
        raise FileNotFoundError(
            f"Statement not found: {STATEMENT}\n"
            "See the README for instructions on obtaining the sample document."
        )
    if not SCHEMA_FILE.exists():
        raise FileNotFoundError(
            f"Schema not found: {SCHEMA_FILE}\n"
            "Run generate_schema.py first, or use the pre-generated activity_schema.json."
        )

    cfg = Config.from_yaml(CONFIG_FILE) if CONFIG_FILE.exists() else Config()
    cfg.extraction_method = EXTRACTION_METHOD

    text = STATEMENT.read_text(encoding="utf-8", errors="replace")
    pages = text.split("\f")

    activity_pages = [
        (i + 1, page)
        for i, page in enumerate(pages)
        if is_activity_page(page)
    ]

    print(f"Extraction method : {EXTRACTION_METHOD}")
    print(f"Activity pages    : {[p for p, _ in activity_pages]}")
    print(f"Output            : {OUTPUT_FILE}\n")

    all_results: dict = {}
    for page_num, page_text in activity_pages:
        label = f"page_{page_num}"
        print(f"  [{label}] {len(page_text):,} chars ...", end=" ", flush=True)
        result = extract(Text(page_text, name=label), schema=SCHEMA_FILE, config=cfg)
        n_fields = len([k for k in result if not k.endswith("_confidence")])
        print(f"{n_fields} top-level fields extracted")
        all_results[label] = result

    OUTPUT_FILE.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nWrote {len(all_results)} page result(s) to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
