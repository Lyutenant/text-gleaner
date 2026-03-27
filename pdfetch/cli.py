from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="pdfetch — structured PDF data extraction via LLM tool calls")

logging.basicConfig(
    format="%(levelname)s %(name)s %(message)s",
    level=logging.INFO,
)


@app.command("generate-schema")
def generate_schema(
    pdfs: Path = typer.Option(..., help="Directory of sample PDFs"),
    description: Path = typer.Option(..., help="Description file (.yaml or .md)"),
    output: Path = typer.Option(Path("schema.json"), help="Output schema JSON path"),
    sample_ratio: float = typer.Option(1.0, help="Fraction of PDFs to sample (0–1)"),
    sample_dir: Optional[Path] = typer.Option(None, help="Explicit curated sample directory"),
) -> None:
    """Phase 1: Generate a JSON extraction schema from sample PDFs and a description."""
    from .schema_generator import generate_schema as _generate_schema

    if not pdfs.exists():
        typer.echo(f"Error: --pdfs path does not exist: {pdfs}", err=True)
        raise typer.Exit(1)
    if not description.exists():
        typer.echo(f"Error: --description file does not exist: {description}", err=True)
        raise typer.Exit(1)

    _generate_schema(
        pdfs_dir=pdfs,
        description_file=description,
        output_file=output,
        sample_ratio=sample_ratio,
        sample_dir=sample_dir,
    )


@app.command("extract")
def extract(
    pdfs: Path = typer.Option(..., help="Directory of PDFs to process"),
    schema: Path = typer.Option(Path("schema.json"), help="Path to schema JSON"),
    output_dir: Optional[Path] = typer.Option(None, help="Directory for output JSON files"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Process first PDF only; print result"),
) -> None:
    """Phase 2: Extract structured data from PDFs using a tool-call schema."""
    from .extractor import extract as _extract

    if not pdfs.exists():
        typer.echo(f"Error: --pdfs path does not exist: {pdfs}", err=True)
        raise typer.Exit(1)
    if not schema.exists():
        typer.echo(f"Error: --schema file does not exist: {schema}", err=True)
        raise typer.Exit(1)

    if not dry_run and output_dir is None:
        output_dir = Path("output")

    _extract(
        pdfs_dir=pdfs,
        schema_file=schema,
        output_dir=output_dir,
        dry_run=dry_run,
    )


if __name__ == "__main__":
    app()
