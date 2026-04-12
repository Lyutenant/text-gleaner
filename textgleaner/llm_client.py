"""Low-level LLM client layer for textgleaner.

This module provides two client classes and a factory function:

- :class:`LLMClient` — talks to a local Ollama server (or any server that
  exposes an OpenAI-compatible ``/v1/chat/completions`` endpoint).
- :class:`ClaudeAPIClient` — talks to Anthropic's hosted Claude API.  Requires
  the ``anthropic`` package (``pip install textgleaner[claude]``).
- :func:`make_client` — factory that returns the right client based on the
  ``provider`` setting.

Both clients expose the same three-method interface:

.. code-block:: python

    response = client.chat(messages, tools=..., tool_choice=..., response_format=...)
    text     = client.get_content(response)      # plain-text reply
    data     = client.get_tool_arguments(response)  # extracted JSON dict

All callers (extractor, schema_generator, schema_refiner) depend only on this
interface, so switching between Ollama and Claude requires no changes in those
modules.

Internally, both clients normalise their server's response into the same
OpenAI-compatible dict shape before returning it, so ``get_content()`` and
``get_tool_arguments()`` are identical for both.
"""
from __future__ import annotations
import json
import logging
import re
from typing import Any

import httpx

from .config import LLMConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model profiles
# ---------------------------------------------------------------------------

# Each profile maps to extra fields that are merged into the
# /v1/chat/completions request payload.  Only keys present in the profile dict
# are sent; the "default" profile is empty (no extra fields).
#
# Profiles let us apply model-specific workarounds without polluting the main
# chat() logic.  The active profile is resolved in _resolve_profile_payload().
PROFILES: dict[str, dict] = {
    "qwen3": {
        # Qwen3 has an extended "thinking" (chain-of-thought) mode that burns
        # a large token budget before generating content.  On big documents this
        # exhausts max_tokens before the actual answer.  Setting think=False
        # disables it.  This field is silently ignored by non-Qwen3 models.
        "extra_body": {"think": False},
    },
    "default": {},
}


def _auto_detect_profile(model: str) -> str:
    """Infer the best profile name from the model name string.

    Currently maps model names containing ``"qwen3"`` to the ``"qwen3"``
    profile and everything else to ``"default"``.

    Args:
        model: The model name as passed to the server (e.g. ``"qwen3:30b"``).

    Returns:
        A profile name key from :data:`PROFILES`.
    """
    name = model.lower()
    if "qwen3" in name:
        return "qwen3"
    return "default"


def _resolve_profile_payload(model: str, profile: str | None) -> dict:
    """Return the extra payload fields for the given model and profile.

    If *profile* is ``None``, the profile is auto-detected from the model name
    via :func:`_auto_detect_profile`.  An unknown profile name raises
    :exc:`ValueError` so configuration mistakes are caught early.

    Args:
        model: Model name string (used for auto-detection when profile is None).
        profile: Explicit profile name, or ``None`` to auto-detect.

    Returns:
        A shallow copy of the profile's extra payload dict (may be empty).

    Raises:
        ValueError: If *profile* is not ``None`` and not in :data:`PROFILES`.
    """
    if not profile:
        profile = _auto_detect_profile(model)
    if profile not in PROFILES:
        raise ValueError(
            f"Unknown model_profile '{profile}'. "
            f"Valid profiles: {sorted(PROFILES)}"
        )
    logger.debug("model=%s profile=%s", model, profile)
    return dict(PROFILES[profile])


# ---------------------------------------------------------------------------
# Ollama / OpenAI-compatible client
# ---------------------------------------------------------------------------

class LLMClient:
    """HTTP client for local Ollama (or any OpenAI-compatible) LLM server.

    Sends requests to ``{base_url}/v1/chat/completions`` using the
    OpenAI-compatible API format.  All requests use ``stream=True`` so the
    TCP connection stays alive during inference — without streaming, Ollama
    buffers the entire response server-side and the connection can time out
    before the first byte arrives (especially over VPN or Tailscale).

    The SSE stream is reassembled into the same response dict shape as a
    non-streaming call, so :meth:`get_content` and :meth:`get_tool_arguments`
    do not need to know whether streaming was used.

    Args:
        base_url: Ollama server base URL (default from ``TEXTGLEANER__LLM__BASE_URL``
            env var, or ``"http://localhost:11434"``).
        model: Model tag to request (e.g. ``"qwen3:30b"``).
        api_key: Auth token; Ollama accepts any non-empty string.
        temperature: Sampling temperature (0–1).
        max_tokens: Maximum tokens to generate per call.
        timeout: HTTP request timeout in seconds.
        model_profile: Named profile for model-specific payload fields.
            ``None`` = auto-detect from model name.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
        model_profile: str | None = None,
    ):
        defaults = LLMConfig()
        self.base_url = (base_url or defaults.base_url).rstrip("/")
        self.model = model or defaults.model
        self.api_key = api_key or defaults.api_key
        self.temperature = temperature if temperature is not None else defaults.temperature
        self.max_tokens = max_tokens if max_tokens is not None else defaults.max_tokens
        self.timeout = timeout if timeout is not None else defaults.timeout_seconds
        # None means "auto-detect from model name at call time"; an explicit
        # string locks the profile and skips auto-detection.
        self.model_profile = model_profile if model_profile is not None else defaults.model_profile

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: dict | str | None = None,
        response_format: dict | None = None,
    ) -> dict:
        """Send a chat request and return a normalised response dict.

        Builds an OpenAI-compatible ``/v1/chat/completions`` payload, streams
        the response via SSE, reassembles the content and tool-call fragments,
        then returns a dict with the same shape as a non-streaming response::

            {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": "<text or None>",
                        "tool_calls": [...]   # only present when the model called a tool
                    },
                    "finish_reason": "stop" | "tool_calls" | ...
                }]
            }

        Args:
            messages: Conversation history in OpenAI message format
                (list of ``{"role": ..., "content": ...}`` dicts).  A
                ``"system"`` role message is supported.
            tools: Optional list of tool definitions in OpenAI format::

                    [{"type": "function", "function": {
                        "name": "...", "description": "...", "parameters": {...}
                    }}]

            tool_choice: How to select a tool.  Pass
                ``{"type": "function", "function": {"name": "..."}}`` to force
                a specific tool call, ``"auto"`` to let the model decide, or
                ``None`` to disable tools even when ``tools`` is provided.
            response_format: Grammar-constrained output format.  Pass
                ``{"type": "json_schema", "json_schema": {"name": "...",
                "schema": {...}, "strict": True}}`` to force JSON output that
                matches the schema (``structured_output`` extraction method).

        Returns:
            A normalised response dict (same shape described above).

        Raises:
            httpx.HTTPStatusError: If the server returns a non-2xx status.
        """
        profile_payload = _resolve_profile_payload(self.model, self.model_profile)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,   # streaming keeps the connection alive over slow/remote links
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            **profile_payload,
        }
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if response_format is not None:
            payload["response_format"] = response_format

        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        logger.debug("POST %s model=%s", url, self.model)

        with httpx.Client(timeout=self.timeout) as client:
            with client.stream("POST", url, json=payload, headers=headers) as resp:
                if not resp.is_success:
                    body = resp.read()
                    logger.error("HTTP %d from %s: %s", resp.status_code, url, body[:500])
                    resp.raise_for_status()

                # Accumulate streaming fragments.
                # content_parts: text tokens from delta.content
                # tool_call_parts: indexed by tool call index; each entry
                #   accumulates the name and arguments JSON string fragment by
                #   fragment as the model generates them.
                content_parts: list[str] = []
                tool_call_parts: dict[int, dict] = {}
                finish_reason: str | None = None

                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    choice = chunk.get("choices", [{}])[0]
                    delta = choice.get("delta", {})
                    finish_reason = choice.get("finish_reason") or finish_reason

                    if delta.get("content"):
                        content_parts.append(delta["content"])

                    for tc in delta.get("tool_calls", []):
                        idx = tc.get("index", 0)
                        if idx not in tool_call_parts:
                            tool_call_parts[idx] = {
                                "id": tc.get("id", ""),
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            }
                        fn = tc.get("function", {})
                        if fn.get("name"):
                            tool_call_parts[idx]["function"]["name"] += fn["name"]
                        if fn.get("arguments"):
                            tool_call_parts[idx]["function"]["arguments"] += fn["arguments"]

        # Reassemble into the same shape as a non-streaming response so callers
        # (get_content / get_tool_arguments) don't need to change.
        content = "".join(content_parts) or None
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_call_parts:
            message["tool_calls"] = [tool_call_parts[i] for i in sorted(tool_call_parts)]

        return {"choices": [{"message": message, "finish_reason": finish_reason}]}

    def get_content(self, response: dict) -> str:
        """Extract the plain-text content from a :meth:`chat` response.

        Returns an empty string if the model produced no text content (e.g.
        when it responded with a tool call instead).

        Args:
            response: The dict returned by :meth:`chat`.

        Returns:
            The assistant's text reply, or ``""`` if absent.
        """
        return response["choices"][0]["message"].get("content") or ""

    def get_tool_arguments(self, response: dict) -> dict:
        """Extract the tool call arguments from a :meth:`chat` response.

        Normally the model responds with a ``tool_calls`` entry whose
        ``function.arguments`` is a JSON string.  However, some Ollama models
        ignore ``tool_choice`` and embed the JSON directly in the ``content``
        field instead.  This method handles both cases transparently.

        Args:
            response: The dict returned by :meth:`chat`.

        Returns:
            The extracted data as a Python dict.

        Raises:
            ValueError: If neither ``tool_calls`` nor parseable ``content`` is
                present in the response.
            json.JSONDecodeError: If the JSON string in ``tool_calls`` or
                ``content`` is malformed.
        """
        tool_calls = response["choices"][0]["message"].get("tool_calls") or []
        if tool_calls:
            args = tool_calls[0]["function"]["arguments"]
            # OpenAI-compatible API returns arguments as a JSON string
            if isinstance(args, str):
                return json.loads(args)
            return args

        # Fallback: model returned JSON in content despite tool_choice being set.
        # This happens occasionally with Qwen3 and other instruction-tuned models
        # that prioritise natural language output over strict tool_choice compliance.
        content = (response["choices"][0]["message"].get("content") or "").strip()
        if content:
            logger.warning("No tool_calls; attempting to parse content as JSON")
            # Strip markdown code fences if present (e.g. ```json ... ```)
            fenced = re.sub(r"^```(?:json)?\s*", "", content)
            fenced = re.sub(r"\s*```$", "", fenced).strip()
            return json.loads(fenced)

        raise ValueError("No tool_calls and no content in response")


# ---------------------------------------------------------------------------
# Claude API client
# ---------------------------------------------------------------------------

class ClaudeAPIClient:
    """Client for Anthropic's hosted Claude API (api.anthropic.com).

    Presents the same three-method interface as :class:`LLMClient`
    (``chat`` / ``get_content`` / ``get_tool_arguments``) so all callers work
    without modification when the provider is switched to ``"claude"``.

    Internally this class translates between OpenAI-format inputs (which the
    rest of the codebase uses) and Anthropic-format API calls:

    - Tools: ``{"type": "function", "function": {...}}``
      → ``{"name": ..., "description": ..., "input_schema": ...}``
    - tool_choice: ``{"type": "function", "function": {"name": "..."}}``
      → ``{"type": "tool", "name": "..."}``
    - response_format (structured output):
      ``{"type": "json_schema", "json_schema": {...}}``
      → ``output_config: {"format": {"type": "json_schema", ...}}``
    - System messages: extracted from the ``messages`` list and passed as the
      top-level ``system=`` parameter (Anthropic's required format).

    After the API call the Anthropic response is converted back into the same
    OpenAI-compatible dict shape that :meth:`get_content` and
    :meth:`get_tool_arguments` expect, so those methods are identical to
    :class:`LLMClient`.

    All requests use streaming (``messages.stream()``) to keep the TCP
    connection alive during long inference runs.

    Requires the ``anthropic`` package::

        pip install anthropic
        # or:
        pip install "textgleaner[claude]"

    Args:
        model: Claude model ID (e.g. ``"claude-opus-4-6"``).  Defaults to
            ``"claude-opus-4-6"`` when not specified.
        api_key: Anthropic API key (``sk-ant-...``).  If the key is the Ollama
            placeholder ``"local"``, it is treated as absent and the
            ``ANTHROPIC_API_KEY`` environment variable is used instead.
        temperature: Sampling temperature (0–1).
        max_tokens: Maximum tokens to generate per call.
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
        **_ignored: Any,
    ):
        try:
            import anthropic as _anthropic
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required to use provider='claude'. "
                "Install it with: pip install anthropic"
            ) from exc

        self._anthropic = _anthropic
        defaults = LLMConfig()
        # Default to a capable Claude model, not the Ollama default.
        self.model = model or "claude-opus-4-6"
        # The Ollama placeholder "local" is meaningless to Anthropic's API.
        # Pass None instead so the SDK falls back to ANTHROPIC_API_KEY env var.
        raw_key = api_key or defaults.api_key
        self.api_key = raw_key if raw_key != "local" else None
        self.temperature = temperature if temperature is not None else defaults.temperature
        self.max_tokens = max_tokens if max_tokens is not None else defaults.max_tokens
        self.timeout = timeout if timeout is not None else defaults.timeout_seconds
        self._client = _anthropic.Anthropic(api_key=self.api_key, timeout=float(self.timeout))

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: dict | str | None = None,
        response_format: dict | None = None,
    ) -> dict:
        """Send a chat request to the Claude API and return a normalised response.

        Translates OpenAI-format inputs to Anthropic format, makes a streaming
        API call, then converts the Anthropic response back to the OpenAI-
        compatible dict shape that the rest of the codebase expects.

        The return value has the same structure as :meth:`LLMClient.chat` — see
        that method's docstring for the full shape.

        Args:
            messages: Conversation history in OpenAI message format.  A
                ``"system"`` role entry is extracted and passed as the
                ``system=`` parameter to the Anthropic API.
            tools: Tool definitions in OpenAI format.  Converted to Anthropic
                format internally (``input_schema`` instead of ``parameters``).
            tool_choice: Tool selection policy.  An OpenAI forced-tool dict
                ``{"type": "function", "function": {"name": "..."}}`` is
                converted to ``{"type": "tool", "name": "..."}``.
            response_format: Structured output schema in OpenAI format.
                Converted to Anthropic's ``output_config`` format internally.

        Returns:
            A normalised response dict (same shape as :meth:`LLMClient.chat`).

        Raises:
            anthropic.APIError: If the Claude API returns an error response.
        """
        # Anthropic requires the system prompt as a separate top-level parameter,
        # not as a message in the conversation list.
        system: str | None = None
        user_messages: list[dict] = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                user_messages.append({"role": msg["role"], "content": msg["content"]})

        # Convert OpenAI tool format → Anthropic format.
        # OpenAI:   {"type": "function", "function": {"name": ..., "parameters": {...}}}
        # Anthropic: {"name": ..., "description": ..., "input_schema": {...}}
        anthropic_tools: list[dict] | None = None
        if tools:
            anthropic_tools = [
                {
                    "name": t["function"]["name"],
                    "description": t["function"].get("description", ""),
                    "input_schema": t["function"]["parameters"],
                }
                for t in tools
            ]

        # Convert OpenAI tool_choice → Anthropic tool_choice.
        # OpenAI:   {"type": "function", "function": {"name": "my_tool"}}
        # Anthropic: {"type": "tool", "name": "my_tool"}
        anthropic_tool_choice: dict | None = None
        if tool_choice:
            if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
                anthropic_tool_choice = {
                    "type": "tool",
                    "name": tool_choice["function"]["name"],
                }
            elif isinstance(tool_choice, str):
                # Passthrough for "auto", "any", "none" string values
                anthropic_tool_choice = {"type": tool_choice}

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": user_messages,
        }
        if system:
            kwargs["system"] = system
        if anthropic_tools:
            kwargs["tools"] = anthropic_tools
        if anthropic_tool_choice:
            kwargs["tool_choice"] = anthropic_tool_choice
        if response_format:
            # Convert OpenAI response_format → Anthropic output_config.
            # OpenAI:   {"type": "json_schema", "json_schema": {"name": ..., "schema": {...}}}
            # Anthropic: {"format": {"type": "json_schema", "json_schema": {...}}}
            js = response_format.get("json_schema", {})
            kwargs["output_config"] = {
                "format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": js.get("name", "response"),
                        "schema": js.get("schema", {}),
                    },
                }
            }

        logger.debug("Claude API model=%s", self.model)

        # Use streaming so the TCP connection stays alive during long inference.
        # get_final_message() blocks until the full response is assembled.
        with self._client.messages.stream(**kwargs) as stream:
            response = stream.get_final_message()

        # Convert Anthropic response content blocks → OpenAI-compatible dict.
        # Anthropic returns a list of typed blocks; we convert each to the
        # shape that get_content() / get_tool_arguments() expect.
        content_text: str | None = None
        tool_calls: list[dict] = []
        for block in response.content:
            if block.type == "text":
                content_text = block.text or None
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        # Anthropic returns the tool input as a Python dict.
                        # Serialize to a JSON string to match the OpenAI format
                        # that get_tool_arguments() expects.
                        "arguments": json.dumps(block.input),
                    },
                })

        message: dict[str, Any] = {"role": "assistant", "content": content_text}
        if tool_calls:
            message["tool_calls"] = tool_calls

        return {
            "choices": [{
                "message": message,
                "finish_reason": response.stop_reason,
            }]
        }

    def get_content(self, response: dict) -> str:
        """Extract the plain-text content from a :meth:`chat` response.

        Returns an empty string if the model responded with only a tool call.

        Args:
            response: The dict returned by :meth:`chat`.

        Returns:
            The assistant's text reply, or ``""`` if absent.
        """
        return response["choices"][0]["message"].get("content") or ""

    def get_tool_arguments(self, response: dict) -> dict:
        """Extract the tool call arguments from a :meth:`chat` response.

        Handles the case where the model returns JSON in ``content`` instead of
        in a proper tool call (same fallback logic as :class:`LLMClient`).

        Args:
            response: The dict returned by :meth:`chat`.

        Returns:
            The extracted data as a Python dict.

        Raises:
            ValueError: If neither ``tool_calls`` nor parseable ``content``
                is present.
            json.JSONDecodeError: If the arguments string is malformed JSON.
        """
        tool_calls = response["choices"][0]["message"].get("tool_calls") or []
        if tool_calls:
            args = tool_calls[0]["function"]["arguments"]
            if isinstance(args, str):
                return json.loads(args)
            return args

        content = (response["choices"][0]["message"].get("content") or "").strip()
        if content:
            logger.warning("No tool_calls; attempting to parse content as JSON")
            fenced = re.sub(r"^```(?:json)?\s*", "", content)
            fenced = re.sub(r"\s*```$", "", fenced).strip()
            return json.loads(fenced)

        raise ValueError("No tool_calls and no content in response")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_client(
    provider: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    timeout: int | None = None,
    model_profile: str | None = None,
) -> LLMClient | ClaudeAPIClient:
    """Create and return the appropriate LLM client for the given provider.

    This is the single point where the provider setting is resolved.  All
    callers (extractor, schema_generator, schema_refiner) call this function
    instead of instantiating :class:`LLMClient` directly, so swapping the
    backend only requires changing ``provider``.

    If *provider* is ``None`` the value is read from :class:`~config.LLMConfig`
    (which checks the ``TEXTGLEANER__LLM__PROVIDER`` env var, defaulting to
    ``"ollama"``).

    Args:
        provider: ``"claude"`` → :class:`ClaudeAPIClient` (requires
            ``anthropic``).  Any other value (``"ollama"``, ``None``, etc.)
            → :class:`LLMClient`.
        base_url: Passed to :class:`LLMClient`; ignored for Claude.
        model: Model name/ID for whichever backend is selected.
        api_key: Auth token.  For Ollama any string works; for Claude this must
            be a real Anthropic API key (or ``None`` to use the env var).
        temperature: Sampling temperature.
        max_tokens: Token generation limit.
        timeout: Request timeout in seconds.
        model_profile: Profile name for :class:`LLMClient`; ignored for Claude.

    Returns:
        A :class:`LLMClient` or :class:`ClaudeAPIClient` instance ready to use.
    """
    effective_provider = provider or LLMConfig().provider
    if effective_provider == "claude":
        return ClaudeAPIClient(
            model=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )
    return LLMClient(
        base_url=base_url,
        model=model,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        model_profile=model_profile,
    )
