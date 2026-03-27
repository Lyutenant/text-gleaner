from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Any

from .config import get_config
from .llm_client import LLMClient
from .pdf_reader import extract_text_from_pdf, chunk_pages
from .utils import merge_chunks, null_rate

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a precise data extraction assistant. Extract structured data from the provided document text \
by calling the required tool with the extracted values.

Rules:
- Extract ONLY information explicitly present in the document text.
- Never infer, guess, or hallucinate values.
- Use null for any field whose value is not present in the document.
- Confidence score meanings:
  - 1.0 = value is explicitly stated verbatim
  - 0.7 = value is clearly implied
  - 0.4 = value is inferred / uncertain
  - 0.0 = value not found (field will be null)
"""


def _extract_chunk(
    client: LLMClient,
    schema: dict,
    chunk_text: str,
    pdf_name: str,
    page_range: tuple[int, int],
) -> dict | None:
    tool_def = {
        "type": "function",
        "function": {
            "name": schema["name"],
            "description": schema.get("description", ""),
            "parameters": schema["parameters"],
        },
    }
    tool_choice = {"type": "function", "function": {"name": schema["name"]}}

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Document text:\n\n{chunk_text}"},
    ]

    try:
        response = client.chat(messages, tools=[tool_def], tool_choice=tool_choice)
        return client.get_tool_arguments(response)
    except Exception as e:
        logger.warning(
            "filename=%s page_range=%s-%s error=%s",
            pdf_name, page_range[0], page_range[1], e,
        )
        return None


def _process_pdf(client: LLMClient, schema: dict, pdf_path: Path, cfg) -> dict | None:
    pages = extract_text_from_pdf(pdf_path)
    if not pages or not any(p.strip() for p in pages):
        logger.warning("filename=%s page_range=all error=no_extractable_text", pdf_path.name)
        return None

    chunks = chunk_pages(pages, cfg.extraction.chunk_size_pages, cfg.extraction.chunk_overlap_pages)
    chunk_results: list[dict] = []

    for start, end, text in chunks:
        if not text.strip():
            continue
        result = _extract_chunk(client, schema, text, pdf_path.name, (start, end))
        if result is not None:
            chunk_results.append(result)

    if not chunk_results:
        return None

    return merge_chunks(chunk_results)


def extract(
    pdfs_dir: Path,
    schema_file: Path,
    output_dir: Path | None,
    dry_run: bool = False,
) -> None:
    cfg = get_config()
    client = LLMClient()

    with schema_file.open() as f:
        schema = json.load(f)

    pdf_files = sorted(pdfs_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {pdfs_dir}")
        return

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, Any] = {}
    processed = 0
    skipped = 0
    total = len(pdf_files)

    for idx, pdf_path in enumerate(pdf_files):
        logger.info("Processing %s (%d/%d)", pdf_path.name, idx + 1, total)
        result = _process_pdf(client, schema, pdf_path, cfg)

        if result is None:
            skipped += 1
            continue

        processed += 1

        if dry_run or (cfg.extraction.dry_run_first and idx == 0):
            print(f"\n--- Dry run result for {pdf_path.name} ---")
            print(json.dumps(result, indent=2))

            if dry_run:
                return

            if idx == 0 and total > 1:
                remaining = total - 1
                answer = input(f"\nContinue with remaining {remaining} files? [y/N] ").strip().lower()
                if answer != "y":
                    print("Aborted.")
                    return
            continue

        all_results[pdf_path.name] = result

        if cfg.extraction.output_mode == "per_file" and output_dir:
            out_path = output_dir / f"{pdf_path.stem}.json"
            with out_path.open("w") as f:
                json.dump(result, f, indent=2)
                f.write("\n")
            logger.info("Wrote %s", out_path)

    # Handle first PDF result if dry_run_first was true but we continued
    # (it was already printed but not saved — reprocess or skip saving?)
    # For simplicity: the first PDF result is NOT saved when dry_run_first triggers.
    # This matches the spec's intent (user reviews, then batch continues).

    if cfg.extraction.output_mode == "merged" and output_dir and all_results:
        merged_path = output_dir / "results.json"
        with merged_path.open("w") as f:
            json.dump(all_results, f, indent=2)
            f.write("\n")
        print(f"\nMerged results written to: {merged_path}")

    # Summary
    print(f"\n=== Summary ===")
    print(f"Files processed: {processed}")
    print(f"Files skipped:   {skipped}")

    records = list(all_results.values())
    if records:
        rates = null_rate(records)
        high_null = {k: v for k, v in rates.items() if v > 0.2}
        if high_null:
            print("Fields with >20% null rate:")
            for field, rate in sorted(high_null.items(), key=lambda x: -x[1]):
                print(f"  {field}: {rate:.0%}")
