from __future__ import annotations
from pathlib import Path
from typing import Union

from .schema_generator import generate_schema as _generate_schema
from .extractor import extract as _extract

PathLike = Union[str, Path]


def generate_schema(
    samples: Union[PathLike, list[PathLike]],
    description: Union[str, PathLike],
    output: Union[PathLike, None] = None,
    *,
    confidence_scores: Union[bool, None] = None,
    base_url: Union[str, None] = None,
    model: Union[str, None] = None,
    api_key: Union[str, None] = None,
    temperature: Union[float, None] = None,
    max_tokens: Union[int, None] = None,
    timeout: Union[int, None] = None,
) -> dict:
    """
    Phase 1: Generate a JSON extraction schema from sample text files.

    Args:
        samples: One or more paths to plain-text sample files.
        description: Either a raw string describing the document type and fields,
                     or a path to a .yaml / .md description file.
        output: Optional path to write the schema JSON. If omitted, schema is
                returned but not saved.
        confidence_scores: Include _confidence sibling fields in the schema.
                           Defaults to TEXTGLEANER__EXTRACTION__CONFIDENCE_SCORES
                           env var, or True.
        base_url: LLM server base URL. Defaults to TEXTGLEANER__LLM__BASE_URL.
        model: Model name. Defaults to TEXTGLEANER__LLM__MODEL.
        api_key: API key. Defaults to TEXTGLEANER__LLM__API_KEY.
        temperature: Sampling temperature. Defaults to TEXTGLEANER__LLM__TEMPERATURE.
        max_tokens: Max tokens to generate. Defaults to TEXTGLEANER__LLM__MAX_TOKENS.
        timeout: Request timeout in seconds. Defaults to TEXTGLEANER__LLM__TIMEOUT_SECONDS.

    Returns:
        The generated schema as a dict.
    """
    if isinstance(samples, (str, Path)):
        samples = [samples]
    sample_paths = [Path(s) for s in samples]
    desc_str = _resolve_description(description)
    out_path = Path(output) if output is not None else None

    return _generate_schema(
        sample_paths,
        desc_str,
        out_path,
        confidence_scores=confidence_scores,
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def extract(
    inputs: Union[PathLike, list[PathLike]],
    schema: Union[dict, PathLike],
    output: Union[PathLike, None] = None,
    max_chars: Union[int, None] = None,
    *,
    base_url: Union[str, None] = None,
    model: Union[str, None] = None,
    api_key: Union[str, None] = None,
    temperature: Union[float, None] = None,
    max_tokens: Union[int, None] = None,
    timeout: Union[int, None] = None,
) -> dict:
    """
    Phase 2: Extract structured data from one or more text files using a schema.

    Args:
        inputs: One or more paths to plain-text files to extract from.
        schema: The extraction schema as a dict, or a path to a schema .json file.
        output: Optional path to write the result JSON.
        max_chars: Max characters per input file before raising an error.
                   Defaults to TEXTGLEANER__EXTRACTION__MAX_CHARS, or 200,000.
                   Set to 0 to disable the limit.
        base_url: LLM server base URL. Defaults to TEXTGLEANER__LLM__BASE_URL.
        model: Model name. Defaults to TEXTGLEANER__LLM__MODEL.
        api_key: API key. Defaults to TEXTGLEANER__LLM__API_KEY.
        temperature: Sampling temperature. Defaults to TEXTGLEANER__LLM__TEMPERATURE.
        max_tokens: Max tokens to generate. Defaults to TEXTGLEANER__LLM__MAX_TOKENS.
        timeout: Request timeout in seconds. Defaults to TEXTGLEANER__LLM__TIMEOUT_SECONDS.

    Returns:
        For a single input: the extracted data dict.
        For multiple inputs: {filename: extracted_data_dict, ...}
    """
    import json as _json

    if isinstance(inputs, (str, Path)):
        single = True
        input_paths = [Path(inputs)]
    else:
        single = False
        input_paths = [Path(i) for i in inputs]

    if isinstance(schema, dict):
        schema_dict = schema
    else:
        with open(schema) as f:
            schema_dict = _json.load(f)

    out_path = Path(output) if output is not None else None

    return _extract(
        input_paths,
        schema_dict,
        out_path,
        single,
        max_chars=max_chars,
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )


def _resolve_description(description: Union[str, PathLike]) -> str:
    import yaml
    p = Path(description) if not isinstance(description, str) else None
    if p is None:
        candidate = Path(description)
        if candidate.exists() and candidate.is_file():
            p = candidate
    if p is not None and p.exists():
        with p.open() as f:
            if p.suffix in (".yaml", ".yml"):
                content = yaml.safe_load(f)
                return yaml.dump(content, default_flow_style=False)
            return f.read()
    return str(description)
