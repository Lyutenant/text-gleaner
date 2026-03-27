from __future__ import annotations
import json
import logging
from typing import Any

import httpx

from .config import get_config

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self):
        cfg = get_config().llm
        self.base_url = cfg.base_url.rstrip("/")
        self.model = cfg.model
        self.api_key = cfg.api_key
        self.temperature = cfg.temperature
        self.max_tokens = cfg.max_tokens
        self.timeout = cfg.timeout_seconds

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: dict | str | None = None,
    ) -> dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if tools is not None:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        url = f"{self.base_url}/v1/chat/completions"
        logger.debug("POST %s model=%s", url, self.model)

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.json()

    def get_content(self, response: dict) -> str:
        return response["choices"][0]["message"]["content"] or ""

    def get_tool_arguments(self, response: dict) -> dict:
        tool_calls = response["choices"][0]["message"].get("tool_calls", [])
        if not tool_calls:
            raise ValueError("No tool_calls in response")
        raw = tool_calls[0]["function"]["arguments"]
        if isinstance(raw, str):
            return json.loads(raw)
        return raw
