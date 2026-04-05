from __future__ import annotations


class TestLLMConfig:
    def test_defaults(self):
        from textgleaner.config import LLMConfig
        cfg = LLMConfig()
        assert cfg.base_url == "http://localhost:11434"
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


class TestConfig:
    def test_direct_kwargs(self):
        from textgleaner import Config
        cfg = Config(base_url="http://myhost:11434", model="qwen3:30b", temperature=0.5)
        assert cfg.base_url == "http://myhost:11434"
        assert cfg.model == "qwen3:30b"
        assert cfg.temperature == 0.5
        assert cfg.max_tokens is None  # unset values are None

    def test_from_yaml(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            "llm:\n"
            "  base_url: http://yaml-host:11434\n"
            "  model: qwen3:30b\n"
            "  api_key: secret\n"
            "  temperature: 0.1\n"
            "  max_tokens: 8192\n"
            "  timeout_seconds: 600\n"
            "extraction:\n"
            "  confidence_scores: false\n"
            "  max_chars: 50000\n"
        )
        from textgleaner import Config
        cfg = Config.from_yaml(yaml_file)
        assert cfg.base_url == "http://yaml-host:11434"
        assert cfg.model == "qwen3:30b"
        assert cfg.api_key == "secret"
        assert cfg.temperature == 0.1
        assert cfg.max_tokens == 8192
        assert cfg.timeout == 600
        assert cfg.confidence_scores is False
        assert cfg.max_chars == 50000

    def test_from_yaml_missing_file_raises(self, tmp_path):
        from textgleaner import Config
        import pytest
        with pytest.raises(FileNotFoundError):
            Config.from_yaml(tmp_path / "nonexistent.yaml")

    def test_from_yaml_partial(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("llm:\n  base_url: http://partial:9999\n")
        from textgleaner import Config
        cfg = Config.from_yaml(yaml_file)
        assert cfg.base_url == "http://partial:9999"
        assert cfg.model is None  # not set in YAML

    def test_config_kwarg_on_extract(self, tmp_path):
        from unittest.mock import MagicMock, patch
        from textgleaner import Config, extract, Text
        cfg = Config(base_url="http://cfg-host:11434", model="cfg-model")

        mock_client = MagicMock()
        mock_client.chat.return_value = {}
        mock_client.get_tool_arguments.return_value = {"value": "42"}

        schema = {
            "name": "extract_data",
            "description": "Test",
            "parameters": {"type": "object", "properties": {
                "value": {"type": ["string", "null"], "description": "Value"},
            }},
        }

        with patch("textgleaner.extractor.LLMClient", return_value=mock_client) as MockClient:
            extract(Text("Value: 42"), schema=schema, config=cfg)

        _, kwargs = MockClient.call_args
        assert kwargs.get("base_url") == "http://cfg-host:11434"
        assert kwargs.get("model") == "cfg-model"

    def test_explicit_kwarg_overrides_config(self, tmp_path):
        from unittest.mock import MagicMock, patch
        from textgleaner import Config, extract, Text
        cfg = Config(base_url="http://cfg-host:11434")

        mock_client = MagicMock()
        mock_client.chat.return_value = {}
        mock_client.get_tool_arguments.return_value = {"value": "x"}

        schema = {
            "name": "extract_data",
            "description": "Test",
            "parameters": {"type": "object", "properties": {
                "value": {"type": ["string", "null"], "description": "Value"},
            }},
        }

        with patch("textgleaner.extractor.LLMClient", return_value=mock_client) as MockClient:
            extract(Text("Value: x"), schema=schema, config=cfg, base_url="http://override:9999")

        _, kwargs = MockClient.call_args
        assert kwargs.get("base_url") == "http://override:9999"


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


class TestModelProfiles:
    def test_auto_detect_qwen3(self):
        from textgleaner.llm_client import _auto_detect_profile
        assert _auto_detect_profile("qwen3:30b") == "qwen3"
        assert _auto_detect_profile("Qwen3-235B") == "qwen3"

    def test_auto_detect_default_for_other_models(self):
        from textgleaner.llm_client import _auto_detect_profile
        assert _auto_detect_profile("llama3:8b") == "default"
        assert _auto_detect_profile("mistral:7b") == "default"
        assert _auto_detect_profile("phi4") == "default"

    def test_resolve_qwen3_includes_extra_body(self):
        from textgleaner.llm_client import _resolve_profile_payload
        payload = _resolve_profile_payload("qwen3:30b", None)
        assert "extra_body" in payload
        assert payload["extra_body"] == {"think": False}

    def test_resolve_default_no_extra_body(self):
        from textgleaner.llm_client import _resolve_profile_payload
        payload = _resolve_profile_payload("llama3:8b", None)
        assert payload == {}

    def test_explicit_profile_overrides_auto_detect(self):
        from textgleaner.llm_client import _resolve_profile_payload
        # Qwen3 model name, but forced to "default" profile
        payload = _resolve_profile_payload("qwen3:30b", "default")
        assert "extra_body" not in payload

    def test_explicit_qwen3_profile_on_non_qwen3_model(self):
        from textgleaner.llm_client import _resolve_profile_payload
        # Non-Qwen3 model name, but forced to "qwen3" profile
        payload = _resolve_profile_payload("llama3:8b", "qwen3")
        assert "extra_body" in payload

    def test_unknown_profile_raises(self):
        import pytest
        from textgleaner.llm_client import _resolve_profile_payload
        with pytest.raises(ValueError, match="Unknown model_profile"):
            _resolve_profile_payload("some-model", "nonexistent")

    def test_qwen3_chat_payload_includes_extra_body(self):
        """Integration: LLMClient built with a qwen3 model sends extra_body in payload."""
        import httpx
        from unittest.mock import patch, MagicMock
        from textgleaner.llm_client import LLMClient

        captured = {}

        def fake_stream(method, url, json=None, headers=None):
            captured["payload"] = json
            # Return a minimal SSE response
            mock_resp = MagicMock()
            mock_resp.is_success = True
            mock_resp.iter_lines.return_value = ['data: [DONE]']
            mock_resp.__enter__ = lambda s: mock_resp
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        client = LLMClient(
            base_url="http://localhost:11434",
            model="qwen3:30b",
            api_key="local",
        )
        with patch.object(client, "chat", wraps=client.chat):
            with patch("httpx.Client") as MockHttpxClient:
                mock_httpx = MagicMock()
                mock_httpx.stream.side_effect = fake_stream
                mock_httpx.__enter__ = lambda s: mock_httpx
                mock_httpx.__exit__ = MagicMock(return_value=False)
                MockHttpxClient.return_value = mock_httpx
                client.chat([{"role": "user", "content": "hi"}])

        assert "extra_body" in captured["payload"]
        assert captured["payload"]["extra_body"] == {"think": False}

    def test_default_chat_payload_excludes_extra_body(self):
        """Integration: LLMClient built with a non-qwen3 model does NOT send extra_body."""
        from unittest.mock import patch, MagicMock
        from textgleaner.llm_client import LLMClient

        captured = {}

        def fake_stream(method, url, json=None, headers=None):
            captured["payload"] = json
            mock_resp = MagicMock()
            mock_resp.is_success = True
            mock_resp.iter_lines.return_value = ['data: [DONE]']
            mock_resp.__enter__ = lambda s: mock_resp
            mock_resp.__exit__ = MagicMock(return_value=False)
            return mock_resp

        client = LLMClient(
            base_url="http://localhost:11434",
            model="llama3:8b",
            api_key="local",
        )
        with patch("httpx.Client") as MockHttpxClient:
            mock_httpx = MagicMock()
            mock_httpx.stream.side_effect = fake_stream
            mock_httpx.__enter__ = lambda s: mock_httpx
            mock_httpx.__exit__ = MagicMock(return_value=False)
            MockHttpxClient.return_value = mock_httpx
            client.chat([{"role": "user", "content": "hi"}])

        assert "extra_body" not in captured["payload"]

    def test_model_profile_env_var(self, monkeypatch):
        monkeypatch.setenv("TEXTGLEANER__LLM__MODEL_PROFILE", "default")
        from textgleaner.llm_client import LLMClient
        client = LLMClient(model="qwen3:30b")
        # Env var forces "default" even for qwen3 model name
        assert client.model_profile == "default"

    def test_config_model_profile_kwarg(self):
        from textgleaner import Config
        cfg = Config(model_profile="default")
        assert cfg.model_profile == "default"

    def test_config_from_yaml_model_profile(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            "llm:\n"
            "  model: llama3:8b\n"
            "  model_profile: default\n"
        )
        from textgleaner import Config
        cfg = Config.from_yaml(yaml_file)
        assert cfg.model_profile == "default"
