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

# Each profile maps to extra fields merged into the /v1/chat/completions payload.
# Fields present here are sent to the server; absent keys are not sent at all.
PROFILES: dict[str, dict] = {
    "qwen3": {
        # Disable extended "thinking" mode to prevent token budget exhaustion
        # on large documents. Qwen3-specific — ignored by other models.
        "extra_body": {"think": False},
    },
    "default": {},
}


def _auto_detect_profile(model: str) -> str:
    """Infer a profile name from the model name string."""
    name = model.lower()
    if "qwen3" in name:
        return "qwen3"
    return "default"


def _resolve_profile_payload(model: str, profile: str | None) -> dict:
    """Return the extra payload fields for the given model and profile.

    If *profile* is None, auto-detects from the model name.
    Raises ValueError for unknown profile names.
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


class LLMClient:
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
        # None means "auto-detect from model name"; explicit string overrides.
        self.model_profile = model_profile if model_profile is not None else defaults.model_profile

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: dict | str | None = None,
        response_format: dict | None = None,
    ) -> dict:
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
        return response["choices"][0]["message"].get("content") or ""

    def get_tool_arguments(self, response: dict) -> dict:
        tool_calls = response["choices"][0]["message"].get("tool_calls") or []
        if tool_calls:
            args = tool_calls[0]["function"]["arguments"]
            # OpenAI-compatible API returns arguments as a JSON string
            if isinstance(args, str):
                return json.loads(args)
            return args

        # Fallback: model returned JSON in content despite tool_choice being set
        content = (response["choices"][0]["message"].get("content") or "").strip()
        if content:
            logger.warning("No tool_calls; attempting to parse content as JSON")
            # Strip markdown code fences if present
            fenced = re.sub(r"^```(?:json)?\s*", "", content)
            fenced = re.sub(r"\s*```$", "", fenced).strip()
            return json.loads(fenced)

        raise ValueError("No tool_calls and no content in response")
