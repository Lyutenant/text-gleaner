from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import List, Optional

import typer

app = typer.Typer(help="textgleaner — structured data extraction from text via LLM tool calls")

logging.basicConfig(format="%(levelname)s %(name)s %(message)s", level=logging.INFO)


@app.command("generate-schema")
def generate_schema(
    samples: List[Path] = typer.Option(..., help="One or more sample text files"),
    description: Path = typer.Option(..., help="Description file (.yaml or .md)"),
    output: Path = typer.Option(Path("schema.json"), help="Output schema JSON path"),
    config: Path = typer.Option(Path("config.yaml"), help="Config YAML file", show_default=True),
) -> None:
    """Phase 1: Generate a JSON extraction schema from sample text files."""
    from textgleaner import Config, generate_schema as _gen

    for p in samples:
        if not p.exists():
            typer.echo(f"Error: sample file does not exist: {p}", err=True)
            raise typer.Exit(1)
    if not description.exists():
        typer.echo(f"Error: description file does not exist: {description}", err=True)
        raise typer.Exit(1)

    cfg = Config.from_yaml(config) if config.exists() else Config()

    _gen(
        samples=samples,
        description=description,
        output=output,
        config=cfg,
    )


@app.command("refine-schema")
def refine_schema(
    schema: Path = typer.Option(..., help="Existing schema JSON to refine"),
    samples: List[Path] = typer.Option(..., help="One or more new sample text files"),
    output: Optional[Path] = typer.Option(None, help="Output path (default: overwrites --schema)"),
    config: Path = typer.Option(Path("config.yaml"), help="Config YAML file", show_default=True),
) -> None:
    """Update an existing schema from new sample documents.

    Runs a two-pass refinement: first the LLM compares the new samples against
    the existing schema (gap analysis), then it produces the complete updated
    schema JSON preserving all existing fields.

    By default the refined schema is written back to --schema (in-place update).
    Pass --output to write to a different file instead.
    """
    from textgleaner import Config, refine_schema as _refine

    if not schema.exists():
        typer.echo(f"Error: schema file does not exist: {schema}", err=True)
        raise typer.Exit(1)
    for p in samples:
        if not p.exists():
            typer.echo(f"Error: sample file does not exist: {p}", err=True)
            raise typer.Exit(1)

    cfg = Config.from_yaml(config) if config.exists() else Config()
    out = output if output is not None else schema

    _refine(
        schema=schema,
        samples=samples,
        output=out,
        config=cfg,
    )


@app.command("extract")
def extract(
    inputs: List[Path] = typer.Option([], help="One or more text files to extract from"),
    inputs_dir: Optional[Path] = typer.Option(None, help="Directory of .txt files to extract from"),
    schema: Path = typer.Option(Path("schema.json"), help="Path to schema JSON"),
    output: Optional[Path] = typer.Option(None, help="Output file (.json, .csv, or .xlsx)"),
    report: Optional[Path] = typer.Option(None, help="Write per-field null-rate summary to this CSV"),
    max_chars: Optional[int] = typer.Option(None, help="Override max chars per file"),
    config: Path = typer.Option(Path("config.yaml"), help="Config YAML file", show_default=True),
) -> None:
    """Phase 2: Extract structured data from text files using a schema.

    Input files can be specified individually (--inputs) or as a directory of
    .txt files (--inputs-dir). Output format is inferred from the --output
    file extension: .json (default), .csv, or .xlsx (requires openpyxl).

    Use --report to also write a per-field null-rate summary CSV.
    """
    from textgleaner import Config, extract as _extract, summarize

    if inputs_dir is not None and inputs:
        typer.echo("Error: use --inputs or --inputs-dir, not both", err=True)
        raise typer.Exit(1)

    if inputs_dir is not None:
        if not inputs_dir.is_dir():
            typer.echo(f"Error: not a directory: {inputs_dir}", err=True)
            raise typer.Exit(1)
        inputs = sorted(inputs_dir.glob("*.txt"))
        if not inputs:
            typer.echo(f"Error: no .txt files found in {inputs_dir}", err=True)
            raise typer.Exit(1)

    if not inputs:
        typer.echo("Error: provide --inputs or --inputs-dir", err=True)
        raise typer.Exit(1)

    for p in inputs:
        if not p.exists():
            typer.echo(f"Error: input file does not exist: {p}", err=True)
            raise typer.Exit(1)
    if not schema.exists():
        typer.echo(f"Error: schema file does not exist: {schema}", err=True)
        raise typer.Exit(1)

    cfg = Config.from_yaml(config) if config.exists() else Config()

    result = _extract(
        inputs=inputs,
        schema=schema,
        output=output,
        max_chars=max_chars,
        config=cfg,
    )

    if report is not None:
        results_dict = result if isinstance(result, dict) and not _is_flat(result) else \
                       {inputs[0].name: result} if len(inputs) == 1 else result
        summarize(results_dict, output=report)
        typer.echo(f"Summary written to {report}")

    if output is None:
        typer.echo(json.dumps(result, indent=2))


def _is_flat(d: dict) -> bool:
    """Heuristic: True if this looks like a single extracted dict rather than
    a {filename: dict} multi-result dict. Used to normalise before summarize()."""
    return bool(d) and not all(isinstance(v, dict) for v in d.values())


@app.command("validate")
def validate(
    inputs: List[Path] = typer.Option(..., help="Sample text files to extract from"),
    schema: Path = typer.Option(Path("schema.json"), help="Path to schema JSON"),
    null_threshold: float = typer.Option(0.5, help="Null-rate above which a field is flagged"),
    confidence_threshold: float = typer.Option(0.5, help="Confidence below which a field is flagged"),
    output: Optional[Path] = typer.Option(None, help="Save report as JSON to this path"),
    config: Path = typer.Option(Path("config.yaml"), help="Config YAML file", show_default=True),
) -> None:
    """Dry-run extraction on sample files and report per-field quality.

    Runs extraction on the provided samples and prints a table showing which
    fields are well-populated, high-null, or low-confidence. Use this to iterate
    on your schema before running a full batch extraction.
    """
    from textgleaner import Config, validate as _validate

    for p in inputs:
        if not p.exists():
            typer.echo(f"Error: input file does not exist: {p}", err=True)
            raise typer.Exit(1)
    if not schema.exists():
        typer.echo(f"Error: schema file does not exist: {schema}", err=True)
        raise typer.Exit(1)

    cfg = Config.from_yaml(config) if config.exists() else Config()

    _validate(
        inputs=inputs,
        schema=schema,
        config=cfg,
        null_threshold=null_threshold,
        confidence_threshold=confidence_threshold,
        output=output,
    )


if __name__ == "__main__":
    app()
