"""
Phase 1 — Generate a JSON extraction schema for the Activity section of a
Fidelity monthly brokerage statement.

Uses pages 11–17 as sample input. Those pages cover every Activity subsection:
Securities Bought & Sold, Trades Pending Settlement, Dividends & Income,
Other Activity, Daily Additions & Subtractions, Debit Card, Checking, Fees.

Usage:
    python generate_schema.py

Outputs:
    activity_schema.json  (in this directory)

Requirements:
    - config.yaml in the repo root (copy from config.example.yaml)
    - statement.txt in this directory (see README for how to get it)
"""
from pathlib import Path

from textgleaner import Config, Text, generate_schema

HERE = Path(__file__).parent
STATEMENT = HERE / "statement.txt"
DESCRIPTION = HERE / "description.yaml"
SCHEMA_OUTPUT = HERE / "activity_schema.json"
CONFIG_FILE = HERE.parent.parent / "config.yaml"


def main() -> None:
    if not STATEMENT.exists():
        raise FileNotFoundError(
            f"Statement not found: {STATEMENT}\n"
            "See the README for instructions on obtaining the sample document."
        )

    cfg = Config.from_yaml(CONFIG_FILE) if CONFIG_FILE.exists() else Config()

    text = STATEMENT.read_text(encoding="utf-8", errors="replace")
    pages = text.split("\f")

    # Pages 11–17 (0-based indices 10–16) cover all Activity subsection types
    sample_text = "\f".join(pages[10:17])
    print(f"Sample: pages 11–17  ({len(sample_text):,} chars)")
    print(f"Output: {SCHEMA_OUTPUT}\n")

    generate_schema(
        samples=Text(sample_text, name="activity_pages_11_to_17"),
        description=DESCRIPTION,
        output=SCHEMA_OUTPUT,
        config=cfg,
    )


if __name__ == "__main__":
    main()
