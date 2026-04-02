from __future__ import annotations
import json
import logging
import re
from typing import Any

import httpx

from .config import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
    ):
        defaults = LLMConfig()
        self.base_url = (base_url or defaults.base_url).rstrip("/")
        self.model = model or defaults.model
        self.api_key = api_key or defaults.api_key
        self.temperature = temperature if temperature is not None else defaults.temperature
        self.max_tokens = max_tokens if max_tokens is not None else defaults.max_tokens
        self.timeout = timeout if timeout is not None else defaults.timeout_seconds

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: dict | str | None = None,
    ) -> dict:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "extra_body": {"think": False},
        }
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        logger.debug("POST %s model=%s", url, self.model)

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload, headers=headers)
            if not resp.is_success:
                logger.error("HTTP %d from %s: %s", resp.status_code, url, resp.text[:500])
            resp.raise_for_status()
            return resp.json()

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
