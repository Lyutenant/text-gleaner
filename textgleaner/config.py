"""Configuration classes for textgleaner.

Settings are loaded from three sources in this priority order (highest first):

1. Explicit kwargs passed directly to :func:`generate_schema` / :func:`extract`
2. A :class:`~textgleaner.Config` object passed as ``config=``
3. Environment variables (``TEXTGLEANER__LLM__*``, ``TEXTGLEANER__EXTRACTION__*``)
4. The hard-coded defaults defined in this module

The two ``pydantic-settings`` classes here (:class:`LLMConfig` and
:class:`ExtractionConfig`) are only responsible for layer 3 & 4 — reading env
vars and applying defaults.  The higher-priority layers are handled by
:class:`~textgleaner.Config` (the user-facing wrapper) and the
``_merge_config()`` helper in ``__init__.py``.
"""
from __future__ import annotations
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseSettings):
    """Low-level LLM connection and generation settings.

    Values are read from environment variables with the prefix
    ``TEXTGLEANER__LLM__`` (e.g. ``TEXTGLEANER__LLM__MODEL``).
    They act as defaults; explicit kwargs always win.

    Attributes:
        provider: Which LLM backend to use.  ``"ollama"`` (default) uses the
            local Ollama server via the OpenAI-compatible ``/v1/chat/completions``
            endpoint.  ``"claude"`` uses Anthropic's hosted API and requires the
            ``anthropic`` package (``pip install textgleaner[claude]``).
        base_url: Base URL of the Ollama server.  Only used when
            ``provider="ollama"``; ignored for ``provider="claude"``.
        model: Model name passed to the server.  For Ollama this is the
            ``ollama pull`` tag (e.g. ``"qwen3:30b"``); for Claude this is the
            Anthropic model ID (e.g. ``"claude-opus-4-6"``).
        api_key: Auth token for the LLM server.  Ollama accepts any non-empty
            string (default ``"local"``).  For Claude this must be a real
            Anthropic API key.
        temperature: Sampling temperature (0–1).  Lower values produce more
            deterministic, focused output.  0.2 works well for extraction tasks.
        max_tokens: Hard cap on tokens the model may generate per call.
        timeout_seconds: HTTP request timeout in seconds.  Set high enough to
            survive slow model inference on large documents.
        model_profile: Named profile that injects model-specific payload fields.
            ``None`` (default) means auto-detect from the model name.  Currently
            ``"qwen3"`` and ``"default"`` are recognised; see ``PROFILES`` in
            ``llm_client.py``.
    """

    provider: str = "ollama"  # "ollama" | "claude"
    base_url: str = "http://localhost:11434"
    model: str = "qwen3-235b"
    api_key: str = "local"
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout_seconds: int = 120
    model_profile: Optional[str] = None  # None = auto-detect from model name

    model_config = SettingsConfigDict(env_prefix="TEXTGLEANER__LLM__", extra="ignore")


class ExtractionConfig(BaseSettings):
    """Settings that control Phase 2 extraction behaviour.

    Values are read from environment variables with the prefix
    ``TEXTGLEANER__EXTRACTION__``.

    Attributes:
        confidence_scores: When ``True`` (default), the schema includes
            ``<field>_confidence`` sibling fields (0–1 floats) alongside every
            data field.  These measure how certain the model was:
            1.0 = verbatim, 0.7 = implied, 0.4 = inferred, 0.0 = not found.
        max_chars: Maximum number of characters allowed in a single input before
            raising a ``ValueError``.  Set to 0 to disable the limit.  The
            default (200 000) prevents accidentally sending huge files to the LLM.
        extraction_method: How to force structured output from the model.

            - ``"tool_call"`` (default) — passes the schema as a tool definition
              and sets ``tool_choice`` to force the model to call it.  The model
              responds with a JSON-encoded function call.
            - ``"structured_output"`` — passes the schema as
              ``response_format: {type: json_schema}`` so the inference engine
              applies grammar-constrained decoding.  More reliable on smaller
              models but much slower.
            - ``"auto"`` — tries ``tool_call`` first; falls back to
              ``structured_output`` on ``ValueError``, ``JSONDecodeError``, or
              HTTP 400/422.  Timeouts and HTTP 5xx are always re-raised.

        confidence_retry: When ``True``, a second targeted LLM call is made for
            any field whose ``_confidence`` score is ≤ 0.4 after the initial
            extraction.  The retry uses a narrowed schema containing only those
            weak fields.  A field is updated only if the retry returns a strictly
            higher confidence score, so a failed retry can never make things
            worse.  Off by default because it adds one extra LLM call per
            document.
    """

    confidence_scores: bool = True
    max_chars: int = 200_000   # per-file limit; 0 = no limit
    extraction_method: str = "tool_call"  # tool_call | structured_output | auto
    confidence_retry: bool = False  # retry fields with confidence ≤ 0.4

    model_config = SettingsConfigDict(env_prefix="TEXTGLEANER__EXTRACTION__", extra="ignore")
