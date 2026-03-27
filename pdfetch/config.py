from __future__ import annotations
from pathlib import Path
from typing import Any, ClassVar, Literal, Tuple, Type

import yaml
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


class _EnvPriorityMixin(BaseSettings):
    """Makes env vars take precedence over init kwargs (YAML values)."""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (env_settings, init_settings, dotenv_settings, file_secret_settings)


class LLMConfig(_EnvPriorityMixin):
    base_url: str = "http://localhost:8080"
    model: str = "qwen3-235b"
    api_key: str = "local"
    temperature: float = 0.2
    max_tokens: int = 4096
    timeout_seconds: int = 120

    model_config = SettingsConfigDict(env_prefix="PDFETCH__LLM__", extra="ignore")


class ExtractionConfig(_EnvPriorityMixin):
    confidence_scores: bool = True
    dry_run_first: bool = True
    output_mode: Literal["per_file", "merged"] = "per_file"
    chunk_size_pages: int = 20
    chunk_overlap_pages: int = 1

    model_config = SettingsConfigDict(env_prefix="PDFETCH__EXTRACTION__", extra="ignore")


class AppConfig:
    def __init__(self, config_path: Path | None = None):
        raw: dict = {}
        if config_path is None:
            config_path = Path("config.yaml")
        if config_path.exists():
            with config_path.open() as f:
                raw = yaml.safe_load(f) or {}

        llm_raw = raw.get("llm", {})
        extraction_raw = raw.get("extraction", {})

        self.llm = LLMConfig(**llm_raw)
        self.extraction = ExtractionConfig(**extraction_raw)


_config: AppConfig | None = None


def get_config(config_path: Path | None = None) -> AppConfig:
    global _config
    if _config is None:
        _config = AppConfig(config_path)
    return _config
