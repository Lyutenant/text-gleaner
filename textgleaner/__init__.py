from __future__ import annotations
import json as _json
from pathlib import Path
from typing import Union

from .schema_generator import generate_schema as _generate_schema
from .extractor import extract as _extract
from .reporter import (
    summarize as _summarize,
    write_csv, write_excel, write_summary_csv,
    build_validation_report, format_validation_report,
)

PathLike = Union[str, Path]


class Config:
    """Holds LLM and extraction configuration for textgleaner.

    Values set here take priority over environment variables, and are
    overridden by any explicit kwargs passed directly to :func:`extract`
    or :func:`generate_schema`.

    Usage::

        from textgleaner import Config, extract

        # Load from a YAML file
        cfg = Config.from_yaml("config.yaml")

        # Or set values directly in code
        cfg = Config(base_url="http://myserver:11434", model="qwen3:30b")

        # Pass to functions — replaces 6 individual kwargs
        result = extract("doc.txt", schema=schema, config=cfg)
    """

    def __init__(
        self,
        *,
        base_url: Union[str, None] = None,
        model: Union[str, None] = None,
        api_key: Union[str, None] = None,
        temperature: Union[float, None] = None,
        max_tokens: Union[int, None] = None,
        timeout: Union[int, None] = None,
        confidence_scores: Union[bool, None] = None,
        max_chars: Union[int, None] = None,
        extraction_method: Union[str, None] = None,
        confidence_retry: Union[bool, None] = None,
    ):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.confidence_scores = confidence_scores
        self.max_chars = max_chars
        self.extraction_method = extraction_method
        self.confidence_retry = confidence_retry

    @classmethod
    def from_yaml(cls, path: PathLike) -> "Config":
        """Load configuration from a YAML file.

        The YAML should follow this structure::

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

        Args:
            path: Path to the YAML config file. Raises :exc:`FileNotFoundError`
                  if the file does not exist.
        """
        import yaml
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        with p.open() as f:
            data = yaml.safe_load(f) or {}
        llm = data.get("llm", {})
        ext = data.get("extraction", {})
        return cls(
            base_url=llm.get("base_url"),
            model=llm.get("model"),
            api_key=llm.get("api_key"),
            temperature=llm.get("temperature"),
            max_tokens=llm.get("max_tokens"),
            timeout=llm.get("timeout_seconds"),
            confidence_scores=ext.get("confidence_scores"),
            max_chars=ext.get("max_chars"),
            extraction_method=ext.get("extraction_method"),
            confidence_retry=ext.get("confidence_retry"),
        )


class Text:
    """Wraps a raw text string for use as input to extract() or generate_schema().

    Use this when you want to pass document text directly rather than a file path —
    for example, when feeding a pre-sliced section of a larger document.

    Args:
        content: The plain-text content to extract from or use as a sample.
        name: A human-readable label used as the key in multi-input results and
              in log messages. Defaults to "<text>".

    Example::

        from textgleaner import extract, Text

        pages = document_text.split("\\f")          # split on form-feed / page break
        holdings_text = "\\n".join(pages[4:8])      # pages 5-8

        result = extract(Text(holdings_text, name="holdings"), schema=holdings_schema)
    """

    def __init__(self, content: str, name: str = "<text>"):
        self.content = content
        self.name = name


# Internal type: a resolved (text, label) pair ready for the core functions.
_TextPair = tuple[str, str]


def _merge_config(config: Union[Config, None], **kwargs) -> dict:
    """Merge a Config object with explicit kwargs. Explicit kwargs take priority."""
    merged: dict = {}
    if config is not None:
        for attr in (
            "base_url", "model", "api_key", "temperature",
            "max_tokens", "timeout", "confidence_scores", "max_chars",
            "extraction_method", "confidence_retry",
        ):
            val = getattr(config, attr, None)
            if val is not None:
                merged[attr] = val
    for k, v in kwargs.items():
        if v is not None:
            merged[k] = v
    return merged


def generate_schema(
    samples: Union[PathLike, Text, list[Union[PathLike, Text]]],
    description: Union[str, PathLike],
    output: Union[PathLike, None] = None,
    *,
    config: Union[Config, None] = None,
    confidence_scores: Union[bool, None] = None,
    base_url: Union[str, None] = None,
    model: Union[str, None] = None,
    api_key: Union[str, None] = None,
    temperature: Union[float, None] = None,
    max_tokens: Union[int, None] = None,
    timeout: Union[int, None] = None,
) -> dict:
    """
    Phase 1: Generate a JSON extraction schema from sample documents.

    Args:
        samples: One or more sample documents. Each can be a file path (str or
                 Path) or a :class:`Text` instance containing raw text.
        description: Either a raw string describing the document type and fields,
                     or a path to a .yaml / .md description file.
        output: Optional path to write the schema JSON. If omitted, schema is
                returned but not saved.
        config: A :class:`Config` instance (from ``Config.from_yaml()`` or
                ``Config(...)``). Individual kwargs below override config values.
        confidence_scores: Include _confidence sibling fields in the schema.
                           Overrides config. Defaults to TEXTGLEANER__EXTRACTION__CONFIDENCE_SCORES
                           env var, or True.
        base_url: LLM server base URL. Overrides config. Defaults to TEXTGLEANER__LLM__BASE_URL.
        model: Model name. Overrides config. Defaults to TEXTGLEANER__LLM__MODEL.
        api_key: API key. Overrides config. Defaults to TEXTGLEANER__LLM__API_KEY.
        temperature: Sampling temperature. Overrides config. Defaults to TEXTGLEANER__LLM__TEMPERATURE.
        max_tokens: Max tokens to generate. Overrides config. Defaults to TEXTGLEANER__LLM__MAX_TOKENS.
        timeout: Request timeout in seconds. Overrides config. Defaults to TEXTGLEANER__LLM__TIMEOUT_SECONDS.

    Returns:
        The generated schema as a dict.
    """
    if not isinstance(samples, list):
        samples = [samples]
    sample_pairs = [_resolve_input(s) for s in samples]
    desc_str = _resolve_description(description)
    out_path = Path(output) if output is not None else None

    resolved = _merge_config(
        config,
        confidence_scores=confidence_scores,
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    return _generate_schema(
        sample_pairs,
        desc_str,
        out_path,
        confidence_scores=resolved.get("confidence_scores"),
        base_url=resolved.get("base_url"),
        model=resolved.get("model"),
        api_key=resolved.get("api_key"),
        temperature=resolved.get("temperature"),
        max_tokens=resolved.get("max_tokens"),
        timeout=resolved.get("timeout"),
    )


def extract(
    inputs: Union[PathLike, Text, list[Union[PathLike, Text]]],
    schema: Union[dict, PathLike],
    output: Union[PathLike, None] = None,
    max_chars: Union[int, None] = None,
    *,
    config: Union[Config, None] = None,
    extraction_method: Union[str, None] = None,
    confidence_retry: Union[bool, None] = None,
    base_url: Union[str, None] = None,
    model: Union[str, None] = None,
    api_key: Union[str, None] = None,
    temperature: Union[float, None] = None,
    max_tokens: Union[int, None] = None,
    timeout: Union[int, None] = None,
) -> dict:
    """
    Phase 2: Extract structured data from one or more documents using a schema.

    Args:
        inputs: One or more documents to extract from. Each can be a file path
                (str or Path) or a :class:`Text` instance containing raw text.
        schema: The extraction schema as a dict, or a path to a schema .json file.
        output: Optional path to write the result JSON.
        max_chars: Max characters per input before raising an error. Overrides
                   config. Defaults to TEXTGLEANER__EXTRACTION__MAX_CHARS, or 200,000.
                   Set to 0 to disable the limit.
        config: A :class:`Config` instance (from ``Config.from_yaml()`` or
                ``Config(...)``). Individual kwargs below override config values.
        extraction_method: ``"tool_call"`` (default), ``"structured_output"``, or
                           ``"auto"``. Overrides config.
                           ``tool_call`` — forces a function/tool call; falls back to
                           content JSON if the model ignores tool_choice.
                           ``structured_output`` — uses ``response_format`` with
                           ``json_schema`` (grammar-constrained decoding); works with
                           models that handle tool_choice poorly.
                           ``auto`` — tries ``tool_call`` first; falls back to
                           ``structured_output`` if the model returns unparseable
                           output or the server returns HTTP 400/422.
        base_url: LLM server base URL. Overrides config. Defaults to TEXTGLEANER__LLM__BASE_URL.
        model: Model name. Overrides config. Defaults to TEXTGLEANER__LLM__MODEL.
        api_key: API key. Overrides config. Defaults to TEXTGLEANER__LLM__API_KEY.
        temperature: Sampling temperature. Overrides config. Defaults to TEXTGLEANER__LLM__TEMPERATURE.
        max_tokens: Max tokens to generate. Overrides config. Defaults to TEXTGLEANER__LLM__MAX_TOKENS.
        timeout: Request timeout in seconds. Overrides config. Defaults to TEXTGLEANER__LLM__TIMEOUT_SECONDS.

    Returns:
        For a single input: the extracted data dict.
        For multiple inputs: {name: extracted_data_dict, ...}
    """
    if isinstance(inputs, (str, Path, Text)):
        single = True
        input_pairs = [_resolve_input(inputs)]
    else:
        single = False
        input_pairs = [_resolve_input(i) for i in inputs]

    if isinstance(schema, dict):
        schema_dict = schema
    else:
        with open(schema) as f:
            schema_dict = _json.load(f)

    out_path = Path(output) if output is not None else None
    non_json = out_path is not None and out_path.suffix in (".csv", ".xlsx")

    resolved = _merge_config(
        config,
        max_chars=max_chars,
        extraction_method=extraction_method,
        confidence_retry=confidence_retry,
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )
    # For CSV/Excel we collect results first, then write ourselves.
    # For JSON we let _extract() write directly.
    results = _extract(
        input_pairs,
        schema_dict,
        None if non_json else out_path,
        single,
        max_chars=resolved.get("max_chars"),
        extraction_method=resolved.get("extraction_method"),
        confidence_retry=resolved.get("confidence_retry"),
        base_url=resolved.get("base_url"),
        model=resolved.get("model"),
        api_key=resolved.get("api_key"),
        temperature=resolved.get("temperature"),
        max_tokens=resolved.get("max_tokens"),
        timeout=resolved.get("timeout"),
    )

    if non_json and out_path is not None:
        # Normalise to {name: dict} even for single-input results
        results_dict = results if not single else {input_pairs[0][1]: results}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix == ".csv":
            write_csv(results_dict, out_path)
        else:
            write_excel(results_dict, out_path)

    return results


def summarize(
    results: dict,
    output: Union[PathLike, None] = None,
) -> dict:
    """Compute per-field null-rate and average confidence from extract() results.

    Args:
        results: The dict returned by :func:`extract` for multiple inputs —
                 ``{name: extracted_dict, ...}``.
        output: Optional path to write the summary. A ``.csv`` extension writes
                a CSV file with columns ``field``, ``null_rate``, ``avg_confidence``.

    Returns:
        ``{field_name: {"null_rate": float, "avg_confidence": float | None}, ...}``
    """
    summary = _summarize(results)
    if output is not None:
        p = Path(output)
        p.parent.mkdir(parents=True, exist_ok=True)
        write_summary_csv(summary, p)
    return summary


def validate(
    inputs: Union[PathLike, Text, list[Union[PathLike, Text]]],
    schema: Union[dict, PathLike],
    *,
    config: Union[Config, None] = None,
    null_threshold: float = 0.5,
    confidence_threshold: float = 0.5,
    output: Union[PathLike, None] = None,
    base_url: Union[str, None] = None,
    model: Union[str, None] = None,
    api_key: Union[str, None] = None,
    temperature: Union[float, None] = None,
    max_tokens: Union[int, None] = None,
    timeout: Union[int, None] = None,
) -> dict:
    """Dry-run extraction on sample documents and report per-field quality.

    Runs :func:`extract` on the provided samples, then classifies each schema
    field as OK, high-null, always-null, or low-confidence. Prints a formatted
    table to stdout and returns the full report dict.

    Use this to iterate on your schema before running a full batch extraction.

    Args:
        inputs: One or more sample documents (file paths or :class:`Text` instances).
        schema: Schema dict or path to a schema JSON file.
        config: :class:`Config` instance.
        null_threshold: null_rate above which a field is flagged ``high_null``
                        (default 0.5).
        confidence_threshold: avg_confidence below which a field is flagged
                              ``low_confidence`` (default 0.5).
        output: Optional path to save the report as JSON.
        base_url, model, api_key, temperature, max_tokens, timeout:
            LLM overrides (same as :func:`extract`).

    Returns:
        Report dict with ``"fields"``, ``"counts"``, and threshold values.
    """
    # Normalise to a list so extract() always returns {name: dict}
    if isinstance(inputs, (str, Path, Text)):
        inputs = [inputs]

    results = extract(
        inputs,
        schema,
        config=config,
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
    )

    summary = _summarize(results)
    report = build_validation_report(summary, null_threshold, confidence_threshold)

    n = len(results)
    names = list(results.keys())
    sample_label = ", ".join(names[:3]) + (" …" if n > 3 else "")
    print(f"Samples ({n}): {sample_label}")
    print(f"Thresholds: null > {null_threshold:.0%}  confidence < {confidence_threshold:.0%}\n")
    print(format_validation_report(report))

    if output is not None:
        import json as _json_mod
        p = Path(output)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            _json_mod.dump(report, f, indent=2)
            f.write("\n")
        print(f"\nReport saved to {output}")

    return report


def _resolve_input(item: Union[PathLike, Text]) -> _TextPair:
    """Resolve a file path or Text instance to a (text, name) tuple."""
    if isinstance(item, Text):
        return (item.content, item.name)
    p = Path(item)
    return (p.read_text(encoding="utf-8", errors="replace"), p.name)


def _resolve_description(description: Union[str, PathLike]) -> str:
    import yaml
    p = Path(description) if not isinstance(description, str) else None
    if p is None and "\n" not in description and len(description) < 512:
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
