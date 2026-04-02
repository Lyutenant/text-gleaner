from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import List, Optional

import typer

app = typer.Typer(help="textgleaner — structured data extraction from text via LLM tool calls")

logging.basicConfig(format="%(levelname)s %(name)s %(message)s", level=logging.INFO)


def _load_yaml_config(config_path: Path) -> dict:
    """Load config.yaml if it exists; return empty dict otherwise."""
    if config_path.exists():
        import yaml
        with config_path.open() as f:
            return yaml.safe_load(f) or {}
    return {}


@app.command("generate-schema")
def generate_schema(
    samples: List[Path] = typer.Option(..., help="One or more sample text files"),
    description: Path = typer.Option(..., help="Description file (.yaml or .md)"),
    output: Path = typer.Option(Path("schema.json"), help="Output schema JSON path"),
    config: Path = typer.Option(Path("config.yaml"), help="Config YAML file", show_default=True),
) -> None:
    """Phase 1: Generate a JSON extraction schema from sample text files."""
    from textgleaner import generate_schema as _gen

    for p in samples:
        if not p.exists():
            typer.echo(f"Error: sample file does not exist: {p}", err=True)
            raise typer.Exit(1)
    if not description.exists():
        typer.echo(f"Error: description file does not exist: {description}", err=True)
        raise typer.Exit(1)

    cfg = _load_yaml_config(config)
    llm = cfg.get("llm", {})
    ext = cfg.get("extraction", {})

    _gen(
        samples=samples,
        description=description,
        output=output,
        confidence_scores=ext.get("confidence_scores"),
        base_url=llm.get("base_url"),
        model=llm.get("model"),
        api_key=llm.get("api_key"),
        temperature=llm.get("temperature"),
        max_tokens=llm.get("max_tokens"),
        timeout=llm.get("timeout_seconds"),
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
    from textgleaner import extract as _extract

    for p in inputs:
        if not p.exists():
            typer.echo(f"Error: input file does not exist: {p}", err=True)
            raise typer.Exit(1)
    if not schema.exists():
        typer.echo(f"Error: schema file does not exist: {schema}", err=True)
        raise typer.Exit(1)

    cfg = _load_yaml_config(config)
    llm = cfg.get("llm", {})
    ext = cfg.get("extraction", {})

    result = _extract(
        inputs=inputs,
        schema=schema,
        output=output,
        max_chars=max_chars or ext.get("max_chars"),
        base_url=llm.get("base_url"),
        model=llm.get("model"),
        api_key=llm.get("api_key"),
        temperature=llm.get("temperature"),
        max_tokens=llm.get("max_tokens"),
        timeout=llm.get("timeout_seconds"),
    )
    if output is None:
        typer.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    app()
