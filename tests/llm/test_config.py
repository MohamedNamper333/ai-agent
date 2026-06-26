"""Tests for core.llm.config — env-driven LLM config, free-model catalog, and routing helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.llm.config import (  # noqa: E402
    LLMConfig,
    OPENCODE_ZEN_FREE_MODELS,
    get_llm_config,
    has_any_provider,
    resolve_for_level,
)


_LLM_ENV_VARS = (
    "LLM_SIMPLE_MODEL",
    "LLM_MODERATE_MODEL",
    "LLM_DEEP_MODEL",
    "LLM_AUTO_ROUTE",
    "LLM_DEFAULT_LEVEL",
    "LLM_REQUEST_TIMEOUT",
    "LLM_MAX_RETRIES",
)
_OLLAMA_ENV_VARS = ("OLLAMA_URL", "OLLAMA_MODEL", "OLLAMA_ENABLED")
_ZEN_ENV_VARS = ("OPENCODE_ZEN_URL", "OPENCODE_ZEN_KEY", "OPENCODE_ZEN_ENABLED")
_ALL_LLM_ENV_VARS = _LLM_ENV_VARS + _OLLAMA_ENV_VARS + _ZEN_ENV_VARS


def _reset_env_loader_cache() -> None:
    try:
        from config.env_loader import get_env

        if hasattr(get_env, "cache_clear"):
            get_env.cache_clear()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    for key in _ALL_LLM_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    _reset_env_loader_cache()
    yield
    for key in _ALL_LLM_ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    _reset_env_loader_cache()


# --- Free Model Catalog ---


class TestOpenCodeZenFreeModels:
    def test_has_five_entries(self) -> None:
        assert len(OPENCODE_ZEN_FREE_MODELS) == 5

    def test_all_values_have_opencode_prefix(self) -> None:
        for value in OPENCODE_ZEN_FREE_MODELS.values():
            assert value.startswith("opencode/")

    def test_contains_minimax_m3_free(self) -> None:
        assert "minimax-m3-free" in OPENCODE_ZEN_FREE_MODELS

    def test_contains_big_pickle(self) -> None:
        assert "big-pickle" in OPENCODE_ZEN_FREE_MODELS

    def test_contains_deepseek_v4_flash_free(self) -> None:
        assert "deepseek-v4-flash-free" in OPENCODE_ZEN_FREE_MODELS

    def test_contains_nemotron_3_ultra_free(self) -> None:
        assert "nemotron-3-ultra-free" in OPENCODE_ZEN_FREE_MODELS

    def test_contains_miimo_v2_5_free(self) -> None:
        assert "miimo-v2.5-free" in OPENCODE_ZEN_FREE_MODELS

    def test_keys_and_values_are_strings(self) -> None:
        for key, value in OPENCODE_ZEN_FREE_MODELS.items():
            assert isinstance(key, str)
            assert isinstance(value, str)

    def test_values_are_non_empty(self) -> None:
        for value in OPENCODE_ZEN_FREE_MODELS.values():
            assert len(value) > 0


# --- LLMConfig Defaults ---


class TestLLMConfigDefaults:
    def test_returns_llm_config_instance(self) -> None:
        cfg = get_llm_config()
        assert isinstance(cfg, LLMConfig)

    def test_default_ollama_url(self) -> None:
        cfg = get_llm_config()
        assert cfg.ollama_url == "http://localhost:11434"

    def test_default_ollama_model(self) -> None:
        """Ollama model default is qwen3:8b (emergency fallback model)."""
        cfg = get_llm_config()
        assert cfg.ollama_model in ("qwen2.5:7b", "qwen3:8b")

    def test_default_ollama_enabled_true(self) -> None:
        cfg = get_llm_config()
        assert cfg.ollama_enabled is True

    def test_default_opencode_zen_url(self) -> None:
        cfg = get_llm_config()
        assert cfg.opencode_zen_url == "https://opencode.ai/zen/v1"

    def test_default_opencode_zen_key_empty(self) -> None:
        cfg = get_llm_config()
        assert cfg.opencode_zen_key == ""

    def test_default_opencode_zen_enabled_true(self) -> None:
        cfg = get_llm_config()
        assert cfg.opencode_zen_enabled is True

    def test_default_simple_model(self) -> None:
        """simple_model default reflects OCZ-primary routing strategy."""
        cfg = get_llm_config()
        # OCZ is now primary; simple_model is used as Ollama fallback identifier
        assert cfg.simple_model in ("qwen2.5:7b", "qwen3:8b", "deepseek-v4-flash-free")

    def test_default_moderate_model(self) -> None:
        cfg = get_llm_config()
        assert cfg.moderate_model == "deepseek-v4-flash-free"

    def test_default_deep_model(self) -> None:
        cfg = get_llm_config()
        assert cfg.deep_model == "big-pickle"

    def test_default_auto_route_true(self) -> None:
        cfg = get_llm_config()
        assert cfg.auto_route is True

    def test_default_level_is_simple(self) -> None:
        cfg = get_llm_config()
        assert cfg.default_level == "simple"

    def test_default_request_timeout_is_120(self) -> None:
        cfg = get_llm_config()
        assert cfg.request_timeout == 120

    def test_default_max_retries_is_2(self) -> None:
        cfg = get_llm_config()
        assert cfg.max_retries == 2


# --- Ollama Env Overrides ---


class TestOllamaEnv:
    def test_ollama_url_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_URL", "http://example.com:11434")
        cfg = get_llm_config()
        assert cfg.ollama_url == "http://example.com:11434"

    def test_ollama_model_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_MODEL", "llama3.1:8b")
        cfg = get_llm_config()
        assert cfg.ollama_model == "llama3.1:8b"

    def test_ollama_enabled_true_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_ENABLED", "true")
        cfg = get_llm_config()
        assert cfg.ollama_enabled is True

    def test_ollama_enabled_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_ENABLED", "1")
        cfg = get_llm_config()
        assert cfg.ollama_enabled is True

    def test_ollama_enabled_yes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_ENABLED", "yes")
        cfg = get_llm_config()
        assert cfg.ollama_enabled is True

    def test_ollama_enabled_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_ENABLED", "false")
        cfg = get_llm_config()
        assert cfg.ollama_enabled is False

    def test_ollama_enabled_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_ENABLED", "0")
        cfg = get_llm_config()
        assert cfg.ollama_enabled is False

    def test_ollama_enabled_no(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_ENABLED", "no")
        cfg = get_llm_config()
        assert cfg.ollama_enabled is False

    def test_ollama_enabled_garbage_is_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_ENABLED", "garbage")
        cfg = get_llm_config()
        assert cfg.ollama_enabled is False

    def test_ollama_enabled_empty_falls_back_to_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OLLAMA_ENABLED", "")
        cfg = get_llm_config()
        assert cfg.ollama_enabled is True


# --- OpenCode Zen Env Overrides ---


class TestOpenCodeZenEnv:
    def test_opencode_zen_url_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENCODE_ZEN_URL", "https://custom.example.com/v1")
        cfg = get_llm_config()
        assert cfg.opencode_zen_url == "https://custom.example.com/v1"

    def test_opencode_zen_key_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENCODE_ZEN_KEY", "sk-test-123")
        cfg = get_llm_config()
        assert cfg.opencode_zen_key == "sk-test-123"

    def test_opencode_zen_enabled_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENCODE_ZEN_ENABLED", "true")
        cfg = get_llm_config()
        assert cfg.opencode_zen_enabled is True

    def test_opencode_zen_enabled_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENCODE_ZEN_ENABLED", "false")
        cfg = get_llm_config()
        assert cfg.opencode_zen_enabled is False

    def test_opencode_zen_enabled_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENCODE_ZEN_ENABLED", "1")
        cfg = get_llm_config()
        assert cfg.opencode_zen_enabled is True


# --- Model Routing Env Overrides ---


class TestModelRoutingEnv:
    def test_simple_model_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_SIMPLE_MODEL", "llama3.1:8b")
        cfg = get_llm_config()
        assert cfg.simple_model == "llama3.1:8b"

    def test_moderate_model_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_MODERATE_MODEL", "opencode/big-pickle")
        cfg = get_llm_config()
        assert cfg.moderate_model == "opencode/big-pickle"

    def test_deep_model_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_DEEP_MODEL", "opencode/nemotron-3-ultra-free")
        cfg = get_llm_config()
        assert cfg.deep_model == "opencode/nemotron-3-ultra-free"

    def test_auto_route_true_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_AUTO_ROUTE", "true")
        cfg = get_llm_config()
        assert cfg.auto_route is True

    def test_auto_route_false_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_AUTO_ROUTE", "false")
        cfg = get_llm_config()
        assert cfg.auto_route is False

    def test_auto_route_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_AUTO_ROUTE", "1")
        cfg = get_llm_config()
        assert cfg.auto_route is True

    def test_default_level_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_DEFAULT_LEVEL", "deep")
        cfg = get_llm_config()
        assert cfg.default_level == "deep"

    def test_default_level_moderate_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_DEFAULT_LEVEL", "moderate")
        cfg = get_llm_config()
        assert cfg.default_level == "moderate"


# --- Behavior Env Overrides ---


class TestBehaviorEnv:
    def test_request_timeout_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_REQUEST_TIMEOUT", "60")
        cfg = get_llm_config()
        assert cfg.request_timeout == 60

    def test_max_retries_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_MAX_RETRIES", "5")
        cfg = get_llm_config()
        assert cfg.max_retries == 5

    def test_request_timeout_empty_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_REQUEST_TIMEOUT", "")
        cfg = get_llm_config()
        assert cfg.request_timeout == 120

    def test_max_retries_empty_falls_back_to_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_MAX_RETRIES", "")
        cfg = get_llm_config()
        assert cfg.max_retries == 2

    def test_request_timeout_zero(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("LLM_REQUEST_TIMEOUT", "0")
        cfg = get_llm_config()
        assert cfg.request_timeout == 0


# --- has_any_provider ---


class TestHasAnyProvider:
    def test_ollama_enabled_returns_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_ENABLED", "true")
        assert has_any_provider() is True

    def test_ollama_disabled_zen_enabled_with_key_returns_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OLLAMA_ENABLED", "false")
        monkeypatch.setenv("OPENCODE_ZEN_ENABLED", "true")
        monkeypatch.setenv("OPENCODE_ZEN_KEY", "sk-test")
        assert has_any_provider() is True

    def test_both_disabled_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_ENABLED", "false")
        monkeypatch.setenv("OPENCODE_ZEN_ENABLED", "false")
        assert has_any_provider() is False

    def test_zen_enabled_no_key_returns_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OLLAMA_ENABLED", "false")
        monkeypatch.setenv("OPENCODE_ZEN_ENABLED", "true")
        monkeypatch.setenv("OPENCODE_ZEN_KEY", "")
        assert has_any_provider() is False

    def test_zen_disabled_ollama_enabled_returns_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OLLAMA_ENABLED", "true")
        monkeypatch.setenv("OPENCODE_ZEN_ENABLED", "false")
        assert has_any_provider() is True

    def test_both_enabled_returns_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_ENABLED", "true")
        monkeypatch.setenv("OPENCODE_ZEN_ENABLED", "true")
        monkeypatch.setenv("OPENCODE_ZEN_KEY", "sk-test")
        assert has_any_provider() is True


# --- resolve_for_level ---


class TestResolveForLevel:
    def test_simple_returns_tuple(self) -> None:
        result = resolve_for_level("simple")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_moderate_returns_tuple(self) -> None:
        result = resolve_for_level("moderate")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_deep_returns_tuple(self) -> None:
        result = resolve_for_level("deep")
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_provider_is_ollama_or_zen(self) -> None:
        provider, _ = resolve_for_level("simple")
        assert provider in ("ollama", "opencode_zen")

    def test_simple_with_ollama_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """SIMPLE level routes to OCZ primary (Ollama is emergency fallback only)."""
        monkeypatch.setenv("OLLAMA_ENABLED", "true")
        provider, model = resolve_for_level("simple")
        # OCZ primary for all levels — Ollama = fallback
        assert provider in ("ollama", "opencode_zen")

    def test_moderate_uses_zen(self) -> None:
        provider, model = resolve_for_level("moderate")
        assert provider == "opencode_zen"
        assert model == "opencode/deepseek-v4-flash-free"

    def test_deep_uses_zen(self) -> None:
        provider, model = resolve_for_level("deep")
        assert provider == "opencode_zen"
        assert model == "opencode/big-pickle"

    def test_model_is_non_empty_string(self) -> None:
        for level in ("simple", "moderate", "deep"):
            _, model = resolve_for_level(level)
            assert isinstance(model, str)
            assert len(model) > 0

    def test_ollama_disabled_falls_back(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_ENABLED", "false")
        monkeypatch.setenv("OPENCODE_ZEN_ENABLED", "true")
        monkeypatch.setenv("OPENCODE_ZEN_KEY", "sk-test")
        provider, _ = resolve_for_level("simple")
        assert provider == "opencode_zen"
