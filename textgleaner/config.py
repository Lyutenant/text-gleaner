from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMConfig(BaseSettings):
    base_url: str = "http://localhost:11434"
    model: str = "qwen3-235b"
    api_key: str = "local"
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout_seconds: int = 120

    model_config = SettingsConfigDict(env_prefix="TEXTGLEANER__LLM__", extra="ignore")


class ExtractionConfig(BaseSettings):
    confidence_scores: bool = True
    max_chars: int = 200_000   # per-file limit; 0 = no limit
    extraction_method: str = "tool_call"  # tool_call | structured_output | auto

    model_config = SettingsConfigDict(env_prefix="TEXTGLEANER__EXTRACTION__", extra="ignore")
