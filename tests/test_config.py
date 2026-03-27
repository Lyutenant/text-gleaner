from __future__ import annotations
import os
from pathlib import Path

import pytest
import yaml


class TestConfigLoading:
    def test_loads_from_yaml(self, tmp_path):
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            "llm:\n  base_url: http://test:9999\n  model: test-model\nextraction:\n  chunk_size_pages: 5\n"
        )
        from pdfetch import config as cfg_module
        cfg_module._config = None  # reset singleton

        app_cfg = cfg_module.AppConfig(config_file)
        assert app_cfg.llm.base_url == "http://test:9999"
        assert app_cfg.llm.model == "test-model"
        assert app_cfg.extraction.chunk_size_pages == 5

    def test_env_var_overrides_yaml(self, tmp_path, monkeypatch):
        config_file = tmp_path / "config.yaml"
        config_file.write_text("llm:\n  base_url: http://yaml:8080\n")

        monkeypatch.setenv("PDFETCH__LLM__BASE_URL", "http://envvar:1234")

        from pdfetch import config as cfg_module
        cfg_module._config = None

        app_cfg = cfg_module.AppConfig(config_file)
        assert app_cfg.llm.base_url == "http://envvar:1234"

    def test_defaults_when_no_config_file(self, tmp_path):
        from pdfetch import config as cfg_module
        cfg_module._config = None

        nonexistent = tmp_path / "no_config.yaml"
        app_cfg = cfg_module.AppConfig(nonexistent)
        # Should use defaults
        assert app_cfg.extraction.output_mode == "per_file"
        assert app_cfg.extraction.chunk_overlap_pages == 1
