"""Tests for core/llm/router.py — LLMRouter drop-in replacement for the legacy LLM class.

Strategy under test (free-tier only):
  SIMPLE   -> Ollama
  MODERATE -> OpenCode Zen with deepseek-v4-flash-free
  DEEP     -> OpenCode Zen with big-pickle
Fallback: primary <-> alternate on LLMError.
"""
from __future__ import annotations

import asyncio
from typing import Any, Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.llm import (
    AllProvidersFailed,
    LLMError,
    LLMRequest,
    LLMResponse,
    ProviderUnavailable,
    ReasoningLevel,
)
from core.llm.config import LLMConfig
from core.llm.router import DEEP_KEYWORDS, MODERATE_KEYWORDS, LLMRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(**overrides: Any) -> LLMConfig:
    """Build an LLMConfig with test-friendly defaults; allow per-test overrides."""
    defaults: dict[str, Any] = dict(
        ollama_url="http://localhost:11434",
        ollama_model="qwen2.5:7b",
        ollama_enabled=True,
        opencode_zen_url="https://opencode.ai/zen/v1",
        opencode_zen_key="test-key",
        opencode_zen_enabled=True,
        simple_model="qwen2.5:7b",
        moderate_model="deepseek-v4-flash-free",
        deep_model="big-pickle",
        auto_route=True,
        default_level=ReasoningLevel.SIMPLE,
        request_timeout=60,
        max_retries=2,
    )
    defaults.update(overrides)
    return LLMConfig(**defaults)


def _make_mock_provider(name: str, response: Any = None) -> MagicMock:
    """Build a mock provider that returns `response` on .generate() and an iterator on .stream()."""
    if response is None or isinstance(response, str):
        response = LLMResponse(
            text=response if isinstance(response, str) else "ok",
            model="qwen2.5:7b",
            provider=name,
            level=ReasoningLevel.SIMPLE,
        )
    provider = MagicMock()
    provider.provider_name = name
    provider.generate.return_value = response
    provider.stream.return_value = iter([response])
    return provider


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

class TestInit:
    def test_default_config_uses_get_llm_config(self) -> None:
        with patch("core.llm.router.get_llm_config") as mock_get:
            mock_get.return_value = _make_config()
            router = LLMRouter()
            assert router.config is mock_get.return_value
            mock_get.assert_called_once()

    def test_custom_config_kept(self) -> None:
        config = _make_config()
        router = LLMRouter(config)
        assert router.config is config

    def test_lazy_providers_start_none(self) -> None:
        router = LLMRouter(_make_config())
        assert router._ollama is None
        assert router._zen is None

    def test_stats_initialized(self) -> None:
        router = LLMRouter(_make_config())
        assert router._stats["by_level"] == {
            ReasoningLevel.SIMPLE: 0,
            ReasoningLevel.MODERATE: 0,
            ReasoningLevel.DEEP: 0,
        }
        assert router._stats["by_provider"] == {"ollama": 0, "opencode_zen": 0}
        assert router._stats["by_model"] == {}
        assert router._stats["fallbacks"] == 0
        assert router._stats["errors"] == 0


# ---------------------------------------------------------------------------
# Properties (lazy provider construction)
# ---------------------------------------------------------------------------

class TestProperties:
    def test_ollama_lazy_build_and_cache(self) -> None:
        with patch("core.llm.ollama_provider.OllamaProvider") as MockOllama:
            MockOllama.return_value = mock_ollama = MagicMock()
            router = LLMRouter(_make_config())
            assert router._ollama is None
            provider = router.ollama
            assert provider is mock_ollama
            MockOllama.assert_called_once()
            assert router.ollama is mock_ollama
            assert MockOllama.call_count == 1

    def test_zen_lazy_build_and_cache(self) -> None:
        with patch("core.llm.opencode_zen_provider.OpenCodeZenProvider") as MockZen:
            MockZen.return_value = mock_zen = MagicMock()
            router = LLMRouter(_make_config())
            assert router._zen is None
            provider = router.zen
            assert provider is mock_zen
            MockZen.assert_called_once()
            assert router.zen is mock_zen
            assert MockZen.call_count == 1

    def test_ollama_disabled_raises(self) -> None:
        config = _make_config(ollama_enabled=False)
        router = LLMRouter(config)
        with pytest.raises(ProviderUnavailable):
            _ = router.ollama

    def test_zen_disabled_raises(self) -> None:
        config = _make_config(opencode_zen_enabled=False)
        router = LLMRouter(config)
        with pytest.raises(ProviderUnavailable):
            _ = router.zen


# ---------------------------------------------------------------------------
# Classify
# ---------------------------------------------------------------------------

class TestClassify:
    def test_auto_route_false_returns_default(self) -> None:
        config = _make_config(auto_route=False, default_level=ReasoningLevel.MODERATE)
        router = LLMRouter(config)
        assert router.classify("anything goes here") == ReasoningLevel.MODERATE

    def test_long_prompt_is_deep(self) -> None:
        router = LLMRouter(_make_config())
        long = " ".join(["word"] * 60)
        assert router.classify(long) == ReasoningLevel.DEEP

    def test_deep_keyword_is_deep(self) -> None:
        router = LLMRouter(_make_config())
        assert router.classify("Please analyze this data") == ReasoningLevel.DEEP
        assert router.classify("Help me design a system") == ReasoningLevel.DEEP

    def test_two_questions_is_deep(self) -> None:
        router = LLMRouter(_make_config())
        assert router.classify("What is X? How does it work?") == ReasoningLevel.DEEP

    def test_four_newlines_is_deep(self) -> None:
        router = LLMRouter(_make_config())
        assert router.classify("line1\nline2\nline3\nline4\nline5") == ReasoningLevel.DEEP

    def test_medium_length_is_moderate(self) -> None:
        router = LLMRouter(_make_config())
        prompt = " ".join(["explain"] * 20)
        assert router.classify(prompt) == ReasoningLevel.MODERATE

    def test_moderate_keyword_is_moderate(self) -> None:
        router = LLMRouter(_make_config())
        assert router.classify("Please summarize this article") == ReasoningLevel.MODERATE
        assert router.classify("What is the capital of France") == ReasoningLevel.MODERATE

    def test_short_prompt_is_simple(self) -> None:
        router = LLMRouter(_make_config())
        assert router.classify("hi") == ReasoningLevel.SIMPLE
        assert router.classify("hello world") == ReasoningLevel.SIMPLE


# ---------------------------------------------------------------------------
# Build Request
# ---------------------------------------------------------------------------

class TestBuildRequest:
    def test_minimal(self) -> None:
        router = LLMRouter(_make_config())
        req = router._build_request("hi")
        assert isinstance(req, LLMRequest)
        assert req.prompt == "hi"
        assert req.system is None
        assert req.level == ReasoningLevel.SIMPLE
        assert req.max_tokens is None
        assert req.temperature is None
        assert req.model_override is None

    def test_with_params(self) -> None:
        router = LLMRouter(_make_config())
        req = router._build_request(
            "hi",
            system="sys",
            level=ReasoningLevel.DEEP,
            max_tokens=100,
            temperature=0.5,
        )
        assert req.system == "sys"
        assert req.level == ReasoningLevel.DEEP
        assert req.max_tokens == 100
        assert req.temperature == 0.5

    def test_auto_classifies_when_no_level(self) -> None:
        router = LLMRouter(_make_config())
        req = router._build_request("Please analyze this in detail")
        assert req.level == ReasoningLevel.DEEP


# ---------------------------------------------------------------------------
# Pick Provider
# ---------------------------------------------------------------------------

class TestPickProvider:
    def test_simple_level_picks_ollama(self) -> None:
        with patch("core.llm.ollama_provider.OllamaProvider") as MockOllama:
            MockOllama.return_value = ollama = MagicMock()
            router = LLMRouter(_make_config())
            req = router._build_request("hi", level=ReasoningLevel.SIMPLE)
            provider = router._pick_provider(req)
            assert provider is ollama

    def test_moderate_level_picks_zen(self) -> None:
        with patch("core.llm.opencode_zen_provider.OpenCodeZenProvider") as MockZen:
            MockZen.return_value = zen = MagicMock()
            router = LLMRouter(_make_config())
            req = router._build_request("hi", level=ReasoningLevel.MODERATE)
            provider = router._pick_provider(req)
            assert provider is zen

    def test_deep_level_picks_zen(self) -> None:
        with patch("core.llm.opencode_zen_provider.OpenCodeZenProvider") as MockZen:
            MockZen.return_value = zen = MagicMock()
            router = LLMRouter(_make_config())
            req = router._build_request("hi", level=ReasoningLevel.DEEP)
            provider = router._pick_provider(req)
            assert provider is zen

    def test_model_override_opencode_prefix(self) -> None:
        with patch("core.llm.opencode_zen_provider.OpenCodeZenProvider") as MockZen:
            MockZen.return_value = zen = MagicMock()
            router = LLMRouter(_make_config())
            req = router._build_request(
                "hi", level=ReasoningLevel.SIMPLE, model_override="opencode/custom-model"
            )
            provider = router._pick_provider(req)
            assert provider is zen

    def test_model_override_zen_model_names(self) -> None:
        with patch("core.llm.opencode_zen_provider.OpenCodeZenProvider") as MockZen:
            MockZen.return_value = zen = MagicMock()
            router = LLMRouter(_make_config())
            for model in (
                "minimax-m3-free",
                "big-pickle",
                "deepseek-v4-flash-free",
                "nemotron-3-ultra-free",
            ):
                req = router._build_request(
                    "hi", level=ReasoningLevel.SIMPLE, model_override=model
                )
                provider = router._pick_provider(req)
                assert provider is zen, f"Model {model!r} should route to zen"

    def test_model_override_ollama_prefix(self) -> None:
        with patch("core.llm.ollama_provider.OllamaProvider") as MockOllama:
            MockOllama.return_value = ollama = MagicMock()
            router = LLMRouter(_make_config())
            req = router._build_request(
                "hi", level=ReasoningLevel.DEEP, model_override="ollama:llama3:8b"
            )
            provider = router._pick_provider(req)
            assert provider is ollama

    def test_model_override_matches_ollama_model(self) -> None:
        with patch("core.llm.ollama_provider.OllamaProvider") as MockOllama:
            MockOllama.return_value = ollama = MagicMock()
            router = LLMRouter(_make_config(ollama_model="llama3:8b"))
            req = router._build_request(
                "hi", level=ReasoningLevel.DEEP, model_override="llama3:8b"
            )
            provider = router._pick_provider(req)
            assert provider is ollama


# ---------------------------------------------------------------------------
# Fallback Provider
# ---------------------------------------------------------------------------

class TestFallbackProvider:
    def test_ollama_falls_back_to_zen(self) -> None:
        with patch("core.llm.opencode_zen_provider.OpenCodeZenProvider") as MockZen:
            MockZen.return_value = zen = MagicMock()
            router = LLMRouter(_make_config())
            ollama = MagicMock()
            ollama.provider_name = "ollama"
            assert router._fallback_provider(ollama) is zen

    def test_zen_falls_back_to_ollama(self) -> None:
        with patch("core.llm.ollama_provider.OllamaProvider") as MockOllama:
            MockOllama.return_value = ollama = MagicMock()
            router = LLMRouter(_make_config())
            zen = MagicMock()
            zen.provider_name = "opencode_zen"
            assert router._fallback_provider(zen) is ollama

    def test_unknown_provider_returns_none(self) -> None:
        router = LLMRouter(_make_config())
        other = MagicMock()
        other.provider_name = "openai"
        assert router._fallback_provider(other) is None


# ---------------------------------------------------------------------------
# Set Provider Model
# ---------------------------------------------------------------------------

class TestSetProviderModel:
    def test_ollama_uses_simple_model(self) -> None:
        router = LLMRouter(_make_config())
        ollama = MagicMock()
        ollama.provider_name = "ollama"
        router._set_provider_model(ollama, ReasoningLevel.SIMPLE)
        assert ollama.model == "qwen2.5:7b"

    def test_zen_uses_moderate_model(self) -> None:
        router = LLMRouter(_make_config())
        zen = MagicMock()
        router._set_provider_model(zen, ReasoningLevel.MODERATE)
        assert zen.model == "deepseek-v4-flash-free"

    def test_zen_uses_deep_model(self) -> None:
        router = LLMRouter(_make_config())
        zen = MagicMock()
        router._set_provider_model(zen, ReasoningLevel.DEEP)
        assert zen.model == "big-pickle"


# ---------------------------------------------------------------------------
# Generate (text path)
# ---------------------------------------------------------------------------

class TestGenerate:
    def test_happy_path(self) -> None:
        with patch("core.llm.ollama_provider.OllamaProvider") as MockOllama:
            MockOllama.return_value = _make_mock_provider("ollama", "hello")
            router = LLMRouter(_make_config())
            result = router.generate("hi", level=ReasoningLevel.SIMPLE)
            assert result == "hello"

    def test_stream_returns_iterator(self) -> None:
        with patch("core.llm.ollama_provider.OllamaProvider") as MockOllama:
            MockOllama.return_value = ollama = MagicMock()
            ollama.provider_name = "ollama"
            ollama.stream.return_value = iter(["a", "b", "c"])
            router = LLMRouter(_make_config())
            result = router.generate("hi", level=ReasoningLevel.SIMPLE, stream=True)
            chunks = list(result)
            assert chunks == ["a", "b", "c"]

    def test_fallback_on_error(self) -> None:
        with patch("core.llm.ollama_provider.OllamaProvider") as MockOllama, \
                patch("core.llm.opencode_zen_provider.OpenCodeZenProvider") as MockZen:
            MockOllama.return_value = ollama = MagicMock()
            ollama.provider_name = "ollama"
            ollama.generate.side_effect = LLMError("ollama failed")
            MockZen.return_value = _make_mock_provider("opencode_zen", "from-zen")
            router = LLMRouter(_make_config())
            result = router.generate("hi", level=ReasoningLevel.SIMPLE)
            assert result == "from-zen"
            assert router._stats["fallbacks"] == 1

    def test_all_providers_failed_raises(self) -> None:
        with patch("core.llm.ollama_provider.OllamaProvider") as MockOllama, \
                patch("core.llm.opencode_zen_provider.OpenCodeZenProvider") as MockZen:
            MockOllama.return_value = ollama = MagicMock()
            ollama.provider_name = "ollama"
            ollama.generate.side_effect = LLMError("ollama failed")
            MockZen.return_value = zen = MagicMock()
            zen.provider_name = "opencode_zen"
            zen.generate.side_effect = LLMError("zen failed")
            router = LLMRouter(_make_config())
            with pytest.raises(AllProvidersFailed):
                router.generate("hi", level=ReasoningLevel.SIMPLE)
            assert router._stats["errors"] == 1

    def test_passes_system_and_params(self) -> None:
        with patch("core.llm.ollama_provider.OllamaProvider") as MockOllama:
            MockOllama.return_value = ollama = MagicMock()
            ollama.provider_name = "ollama"
            ollama.generate.return_value = LLMResponse(
                text="ok",
                model="qwen2.5:7b",
                provider="ollama",
                level=ReasoningLevel.SIMPLE,
            )
            router = LLMRouter(_make_config())
            router.generate(
                "hi",
                level=ReasoningLevel.SIMPLE,
                system="sys",
                max_tokens=100,
                temperature=0.5,
            )
            req = ollama.generate.call_args[0][0]
            assert req.system == "sys"
            assert req.max_tokens == 100
            assert req.temperature == 0.5


# ---------------------------------------------------------------------------
# Generate Full (LLMResponse path)
# ---------------------------------------------------------------------------

class TestGenerateFull:
    def test_happy_path(self) -> None:
        with patch("core.llm.ollama_provider.OllamaProvider") as MockOllama:
            MockOllama.return_value = ollama = MagicMock()
            ollama.provider_name = "ollama"
            ollama.generate.return_value = LLMResponse(
                text="hello",
                model="qwen2.5:7b",
                provider="ollama",
                level=ReasoningLevel.SIMPLE,
            )
            router = LLMRouter(_make_config())
            req = LLMRequest(prompt="hi", level=ReasoningLevel.SIMPLE)
            resp = router.generate_full(req)
            assert isinstance(resp, LLMResponse)
            assert resp.text == "hello"

    def test_fallback_on_error(self) -> None:
        with patch("core.llm.ollama_provider.OllamaProvider") as MockOllama, \
                patch("core.llm.opencode_zen_provider.OpenCodeZenProvider") as MockZen:
            MockOllama.return_value = ollama = MagicMock()
            ollama.provider_name = "ollama"
            ollama.generate.side_effect = LLMError("ollama failed")
            MockZen.return_value = zen = MagicMock()
            zen.provider_name = "opencode_zen"
            zen.generate.return_value = LLMResponse(
                text="from-zen",
                model="deepseek-v4-flash-free",
                provider="opencode_zen",
                level=ReasoningLevel.SIMPLE,
            )
            router = LLMRouter(_make_config())
            req = LLMRequest(prompt="hi", level=ReasoningLevel.SIMPLE)
            resp = router.generate_full(req)
            assert resp.text == "from-zen"

    def test_all_failed_raises(self) -> None:
        with patch("core.llm.ollama_provider.OllamaProvider") as MockOllama, \
                patch("core.llm.opencode_zen_provider.OpenCodeZenProvider") as MockZen:
            MockOllama.return_value = ollama = MagicMock()
            ollama.provider_name = "ollama"
            ollama.generate.side_effect = LLMError("ollama failed")
            MockZen.return_value = zen = MagicMock()
            zen.provider_name = "opencode_zen"
            zen.generate.side_effect = LLMError("zen failed")
            router = LLMRouter(_make_config())
            req = LLMRequest(prompt="hi", level=ReasoningLevel.SIMPLE)
            with pytest.raises(AllProvidersFailed):
                router.generate_full(req)


# ---------------------------------------------------------------------------
# Stream
# ---------------------------------------------------------------------------

class TestStream:
    def test_happy_path(self) -> None:
        with patch("core.llm.ollama_provider.OllamaProvider") as MockOllama:
            MockOllama.return_value = ollama = MagicMock()
            ollama.provider_name = "ollama"
            ollama.stream.return_value = iter(["a", "b", "c"])
            router = LLMRouter(_make_config())
            chunks = list(router.stream("hi", level=ReasoningLevel.SIMPLE))
            assert chunks == ["a", "b", "c"]

    def test_stream_falls_back(self) -> None:
        with patch("core.llm.ollama_provider.OllamaProvider") as MockOllama, \
                patch("core.llm.opencode_zen_provider.OpenCodeZenProvider") as MockZen:
            MockOllama.return_value = ollama = MagicMock()
            ollama.provider_name = "ollama"
            ollama.stream.side_effect = LLMError("ollama stream failed")
            MockZen.return_value = zen = MagicMock()
            zen.provider_name = "opencode_zen"
            zen.stream.return_value = iter(["x", "y"])
            router = LLMRouter(_make_config())
            chunks = list(router.stream("hi", level=ReasoningLevel.SIMPLE))
            assert chunks == ["x", "y"]
            assert router._stats["fallbacks"] == 1

    def test_stream_all_failed_raises(self) -> None:
        with patch("core.llm.ollama_provider.OllamaProvider") as MockOllama, \
                patch("core.llm.opencode_zen_provider.OpenCodeZenProvider") as MockZen:
            MockOllama.return_value = ollama = MagicMock()
            ollama.provider_name = "ollama"
            ollama.stream.side_effect = LLMError("ollama stream failed")
            MockZen.return_value = zen = MagicMock()
            zen.provider_name = "opencode_zen"
            zen.stream.side_effect = LLMError("zen stream failed")
            router = LLMRouter(_make_config())
            with pytest.raises(AllProvidersFailed):
                list(router.stream("hi", level=ReasoningLevel.SIMPLE))


# ---------------------------------------------------------------------------
# Async Generate
# ---------------------------------------------------------------------------

class TestAgenerate:
    def test_agenerate_returns_generate_result(self) -> None:
        with patch("core.llm.ollama_provider.OllamaProvider") as MockOllama:
            MockOllama.return_value = _make_mock_provider("ollama", "ok")
            router = LLMRouter(_make_config())
            with patch.object(router, "generate", return_value="ok") as mock_gen:
                result = asyncio.run(router.agenerate("hi", level=ReasoningLevel.SIMPLE))
            mock_gen.assert_called_once_with("hi", level=ReasoningLevel.SIMPLE)
            assert result == "ok"


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_get_stats_includes_config(self) -> None:
        router = LLMRouter(_make_config())
        stats = router.get_stats()
        assert "config" in stats
        assert "by_level" in stats
        assert "by_provider" in stats
        assert "by_model" in stats
        assert "fallbacks" in stats
        assert "errors" in stats

    def test_reset_stats_zeros_counters_keeps_config(self) -> None:
        router = LLMRouter(_make_config())
        router._stats["by_provider"]["ollama"] = 5
        router._stats["errors"] = 3
        original_config = router.config
        router.reset_stats()
        assert router._stats["by_provider"] == {"ollama": 0, "opencode_zen": 0}
        assert router._stats["errors"] == 0
        assert router._stats["by_level"] == {
            ReasoningLevel.SIMPLE: 0,
            ReasoningLevel.MODERATE: 0,
            ReasoningLevel.DEEP: 0,
        }
        assert router.config is original_config

    def test_stats_increment_by_provider_and_model(self) -> None:
        with patch("core.llm.ollama_provider.OllamaProvider") as MockOllama:
            MockOllama.return_value = _make_mock_provider("ollama", "ok")
            router = LLMRouter(_make_config())
            router.generate("hi", level=ReasoningLevel.SIMPLE)
            assert router._stats["by_provider"].get("ollama", 0) >= 1
            assert router._stats["by_model"].get("qwen2.5:7b", 0) >= 1

    def test_stats_increment_by_level(self) -> None:
        with patch("core.llm.ollama_provider.OllamaProvider") as MockOllama:
            MockOllama.return_value = _make_mock_provider("ollama", "ok")
            router = LLMRouter(_make_config())
            router.generate("hi", level=ReasoningLevel.SIMPLE)
            assert router._stats["by_level"][ReasoningLevel.SIMPLE] >= 1


# ---------------------------------------------------------------------------
# Keyword Constants
# ---------------------------------------------------------------------------

class TestKeywordConstants:
    def test_deep_keywords_is_tuple(self) -> None:
        assert isinstance(DEEP_KEYWORDS, tuple)
        assert len(DEEP_KEYWORDS) > 0
        assert "analyze" in DEEP_KEYWORDS
        assert "design" in DEEP_KEYWORDS

    def test_moderate_keywords_is_tuple(self) -> None:
        assert isinstance(MODERATE_KEYWORDS, tuple)
        assert len(MODERATE_KEYWORDS) > 0
        assert "summarize" in MODERATE_KEYWORDS
        assert "explain" in MODERATE_KEYWORDS
