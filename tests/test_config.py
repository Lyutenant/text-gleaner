from __future__ import annotations


class TestLLMConfig:
    def test_defaults(self):
        from textgleaner.config import LLMConfig
        cfg = LLMConfig()
        assert cfg.base_url == "http://localhost:8080"
        assert cfg.model == "qwen3-235b"
        assert cfg.temperature == 0.2
        assert cfg.timeout_seconds == 120

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("TEXTGLEANER__LLM__BASE_URL", "http://envvar:9999")
        monkeypatch.setenv("TEXTGLEANER__LLM__MODEL", "custom-model")
        from textgleaner.config import LLMConfig
        cfg = LLMConfig()
        assert cfg.base_url == "http://envvar:9999"
        assert cfg.model == "custom-model"


class TestExtractionConfig:
    def test_defaults(self):
        from textgleaner.config import ExtractionConfig
        cfg = ExtractionConfig()
        assert cfg.max_chars == 200_000
        assert cfg.confidence_scores is True

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("TEXTGLEANER__EXTRACTION__MAX_CHARS", "50000")
        monkeypatch.setenv("TEXTGLEANER__EXTRACTION__CONFIDENCE_SCORES", "false")
        from textgleaner.config import ExtractionConfig
        cfg = ExtractionConfig()
        assert cfg.max_chars == 50000
        assert cfg.confidence_scores is False


class TestLLMClient:
    def test_kwargs_override_env(self, monkeypatch):
        monkeypatch.setenv("TEXTGLEANER__LLM__BASE_URL", "http://env:8080")
        from textgleaner.llm_client import LLMClient
        client = LLMClient(base_url="http://kwarg:1234")
        assert client.base_url == "http://kwarg:1234"

    def test_falls_back_to_env(self, monkeypatch):
        monkeypatch.setenv("TEXTGLEANER__LLM__BASE_URL", "http://env:5678")
        from textgleaner.llm_client import LLMClient
        client = LLMClient()
        assert client.base_url == "http://env:5678"

    def test_trailing_slash_stripped(self):
        from textgleaner.llm_client import LLMClient
        client = LLMClient(base_url="http://host:8080/")
        assert client.base_url == "http://host:8080"
