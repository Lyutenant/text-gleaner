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


@app.command("extract")
def extract(
    inputs: List[Path] = typer.Option(..., help="One or more text files to extract from"),
    schema: Path = typer.Option(Path("schema.json"), help="Path to schema JSON"),
    output: Optional[Path] = typer.Option(None, help="Output JSON file path"),
    max_chars: Optional[int] = typer.Option(None, help="Override max chars per file"),
    config: Path = typer.Option(Path("config.yaml"), help="Config YAML file", show_default=True),
) -> None:
    """Phase 2: Extract structured data from text files using a schema."""
    from textgleaner import Config, extract as _extract

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
    if output is None:
        typer.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    app()
