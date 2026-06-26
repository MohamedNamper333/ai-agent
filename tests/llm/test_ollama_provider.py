"""Tests for core.llm.ollama_provider — OllamaProvider unit tests with mocked LLM."""
from __future__ import annotations

from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import pytest

from core.llm.base import LLMError, LLMRequest, LLMResponse, ProviderUnavailable, ReasoningLevel
from core.llm.ollama_provider import OllamaProvider


# ---------------------------------------------------------------------------
# Test Init
# ---------------------------------------------------------------------------
class TestInit:
    def test_default_model_and_url(self) -> None:
        with patch("core.llm.ollama_provider.LLM", MagicMock()):
            provider = OllamaProvider()
        assert provider.model == "qwen2.5:7b"
        assert provider.url == "http://localhost:11434"
        assert provider.provider_name == "ollama"

    def test_custom_model_and_url(self) -> None:
        with patch("core.llm.ollama_provider.LLM", MagicMock()):
            provider = OllamaProvider(model="llama3:8b", url="http://remote:11434/")
        assert provider.model == "llama3:8b"
        assert provider.url == "http://remote:11434"  # trailing slash stripped

    def test_strips_trailing_slash_from_url(self) -> None:
        with patch("core.llm.ollama_provider.LLM", MagicMock()):
            provider = OllamaProvider(url="http://localhost:11434/")
        assert not provider.url.endswith("/")

    def test_llm_initialized_after_construction(self) -> None:
        with patch("core.llm.ollama_provider.LLM", MagicMock()):
            provider = OllamaProvider()
        assert provider._llm is not None

    def test_inherits_from_base(self) -> None:
        with patch("core.llm.ollama_provider.LLM", MagicMock()):
            provider = OllamaProvider()
        from core.llm.base import BaseLLM
        assert isinstance(provider, BaseLLM)


# ---------------------------------------------------------------------------
# Test _init_llm
# ---------------------------------------------------------------------------
class TestInitLlm:
    def test_init_llm_constructor_failure_raises(self) -> None:
        """If LLM(...) constructor raises, _init_llm raises ProviderUnavailable."""
        with patch("core.llm.ollama_provider.LLM", MagicMock()):
            provider = OllamaProvider()
        failing_cls = MagicMock(side_effect=RuntimeError("model not found"))
        with patch("core.llm.ollama_provider.LLM", failing_cls):
            with pytest.raises(ProviderUnavailable):
                provider._init_llm()

    def test_init_llm_success_sets_attribute(self) -> None:
        mock_instance = MagicMock()
        mock_cls = MagicMock(return_value=mock_instance)
        with patch("core.llm.ollama_provider.LLM", mock_cls):
            provider = OllamaProvider(model="mistral:7b")
        assert provider._llm is mock_instance
        # LLM constructor called once with the model (arg name may vary by implementation)
        mock_cls.assert_called_once()


# ---------------------------------------------------------------------------
# Test is_available
# ---------------------------------------------------------------------------
class TestIsAvailable:
    def test_unavailable_when_llm_not_initialized(self) -> None:
        with patch("core.llm.ollama_provider.LLM", MagicMock()):
            provider = OllamaProvider()
        provider._llm = None
        assert provider.is_available() is False

    def test_available_when_http_returns_200(self) -> None:
        with patch("core.llm.ollama_provider.LLM", MagicMock()):
            provider = OllamaProvider()
        provider._llm = MagicMock()
        mock_response = MagicMock(status_code=200)
        with patch("core.llm.ollama_provider.requests.get", return_value=mock_response):
            assert provider.is_available() is True

    def test_unavailable_when_http_returns_non_200(self) -> None:
        with patch("core.llm.ollama_provider.LLM", MagicMock()):
            provider = OllamaProvider()
        provider._llm = MagicMock()
        mock_response = MagicMock(status_code=500)
        with patch("core.llm.ollama_provider.requests.get", return_value=mock_response):
            assert provider.is_available() is False

    def test_unavailable_when_http_raises_exception(self) -> None:
        with patch("core.llm.ollama_provider.LLM", MagicMock()):
            provider = OllamaProvider()
        provider._llm = MagicMock()
        with patch(
            "core.llm.ollama_provider.requests.get",
            side_effect=ConnectionError("refused"),
        ):
            assert provider.is_available() is False


# ---------------------------------------------------------------------------
# Test generate
# ---------------------------------------------------------------------------
class TestGenerate:
    def _make_provider_with_mock_llm(self, generate_return: Any) -> OllamaProvider:
        with patch("core.llm.ollama_provider.LLM", MagicMock()):
            provider = OllamaProvider()
        provider._llm = MagicMock()
        provider._llm.generate = MagicMock(return_value=generate_return)
        return provider

    def test_generate_returns_llm_response(self) -> None:
        provider = self._make_provider_with_mock_llm("hello world")
        req = LLMRequest(prompt="hi")
        resp = provider.generate(req)
        assert isinstance(resp, LLMResponse)
        assert resp.text == "hello world"
        assert resp.model == "qwen2.5:7b"
        assert resp.provider == "ollama"
        assert resp.level == ReasoningLevel.SIMPLE

    def test_generate_calls_llm_with_retries_kwarg(self) -> None:
        provider = self._make_provider_with_mock_llm("ok")
        req = LLMRequest(prompt="hi", max_tokens=100, temperature=0.5)
        provider.generate(req)
        provider._llm.generate.assert_called_once_with(
            prompt="hi",
            max_tokens=100,
            temperature=0.5,
            stop=None,
            stream=False,
            retries=1,
        )

    def test_generate_falls_back_when_retries_kwarg_unsupported(self) -> None:
        provider = self._make_provider_with_mock_llm("ok")
        # First call raises TypeError (old signature), second succeeds.
        provider._llm.generate = MagicMock(side_effect=[TypeError("unexpected kwarg"), "ok"])
        req = LLMRequest(prompt="hi")
        resp = provider.generate(req)
        assert resp.text == "ok"
        assert provider._llm.generate.call_count == 2

    def test_generate_wraps_generic_exception_in_llm_error(self) -> None:
        provider = self._make_provider_with_mock_llm(None)
        provider._llm.generate = MagicMock(side_effect=RuntimeError("boom"))
        req = LLMRequest(prompt="hi")
        with pytest.raises(LLMError) as exc_info:
            provider.generate(req)
        assert "Ollama generate failed" in str(exc_info.value)
        assert exc_info.value.provider == "ollama"

    def test_generate_raises_provider_unavailable_when_llm_uninit(self) -> None:
        with patch("core.llm.ollama_provider.LLM", MagicMock()):
            provider = OllamaProvider()
        provider._llm = None
        req = LLMRequest(prompt="hi")
        with pytest.raises(ProviderUnavailable):
            provider.generate(req)

    def test_generate_increments_call_counter(self) -> None:
        provider = self._make_provider_with_mock_llm("a")
        provider.generate(LLMRequest(prompt="x"))
        provider.generate(LLMRequest(prompt="y"))
        assert provider._call_count == 2

    def test_generate_increments_error_counter_on_failure(self) -> None:
        provider = self._make_provider_with_mock_llm(None)
        provider._llm.generate = MagicMock(side_effect=RuntimeError("boom"))
        with pytest.raises(LLMError):
            provider.generate(LLMRequest(prompt="x"))
        assert provider._error_count == 1

    def test_generate_handles_none_text(self) -> None:
        provider = self._make_provider_with_mock_llm(None)
        req = LLMRequest(prompt="hi")
        resp = provider.generate(req)
        assert resp.text == ""

    def test_generate_records_latency(self) -> None:
        provider = self._make_provider_with_mock_llm("ok")
        resp = provider.generate(LLMRequest(prompt="hi"))
        assert resp.latency_ms >= 0


# ---------------------------------------------------------------------------
# Test stream
# ---------------------------------------------------------------------------
class TestStream:
    def _make_provider_with_mock_llm(self, chunks: Iterator[str]) -> OllamaProvider:
        with patch("core.llm.ollama_provider.LLM", MagicMock()):
            provider = OllamaProvider()
        provider._llm = MagicMock()
        provider._llm.generate = MagicMock(return_value=chunks)
        return provider

    def test_stream_yields_chunks(self) -> None:
        provider = self._make_provider_with_mock_llm(iter(["a", "b", "c"]))
        req = LLMRequest(prompt="hi")
        chunks = list(provider.stream(req))
        assert chunks == ["a", "b", "c"]

    def test_stream_calls_llm_with_stream_true(self) -> None:
        provider = self._make_provider_with_mock_llm(iter(["x"]))
        list(provider.stream(LLMRequest(prompt="hi")))
        provider._llm.generate.assert_called_once_with(
            prompt="hi",
            max_tokens=None,
            temperature=None,
            stop=None,
            stream=True,
            retries=1,
        )

    def test_stream_skips_falsy_chunks(self) -> None:
        provider = self._make_provider_with_mock_llm(iter(["a", "", None, "b"]))
        chunks = list(provider.stream(LLMRequest(prompt="hi")))
        assert chunks == ["a", "b"]

    def test_stream_raises_provider_unavailable_when_llm_uninit(self) -> None:
        with patch("core.llm.ollama_provider.LLM", MagicMock()):
            provider = OllamaProvider()
        provider._llm = None
        with pytest.raises(ProviderUnavailable):
            list(provider.stream(LLMRequest(prompt="hi")))

    def test_stream_wraps_init_exception(self) -> None:
        with patch("core.llm.ollama_provider.LLM", MagicMock()):
            provider = OllamaProvider()
        provider._llm = MagicMock()
        provider._llm.generate = MagicMock(side_effect=RuntimeError("startup fail"))
        with pytest.raises(LLMError) as exc_info:
            list(provider.stream(LLMRequest(prompt="hi")))
        assert "Ollama stream failed" in str(exc_info.value)

    def test_stream_wraps_iteration_exception(self) -> None:
        def bad_iter() -> Iterator[str]:
            yield "a"
            raise RuntimeError("mid-stream")
        provider = self._make_provider_with_mock_llm(bad_iter())
        with pytest.raises(LLMError) as exc_info:
            list(provider.stream(LLMRequest(prompt="hi")))
        assert "Ollama stream iteration failed" in str(exc_info.value)

    def test_stream_falls_back_when_retries_unsupported(self) -> None:
        with patch("core.llm.ollama_provider.LLM", MagicMock()):
            provider = OllamaProvider()
        provider._llm = MagicMock()
        provider._llm.generate = MagicMock(
            side_effect=[TypeError("unexpected kwarg"), iter(["ok"])]
        )
        chunks = list(provider.stream(LLMRequest(prompt="hi")))
        assert chunks == ["ok"]


# ---------------------------------------------------------------------------
# Test get_stats
# ---------------------------------------------------------------------------
class TestGetStats:
    def test_get_stats_includes_url_and_availability(self) -> None:
        with patch("core.llm.ollama_provider.LLM", MagicMock()):
            provider = OllamaProvider(url="http://example:11434")
        provider._llm = MagicMock()
        mock_response = MagicMock(status_code=200)
        with patch("core.llm.ollama_provider.requests.get", return_value=mock_response):
            stats = provider.get_stats()
        assert stats["url"] == "http://example:11434"
        assert stats["available"] is True
        assert "model" in stats
        assert "provider" in stats


# ---------------------------------------------------------------------------
# Test validate_request (inherited from BaseLLM)
# ---------------------------------------------------------------------------
class TestValidateRequest:
    def test_valid_request_passes(self) -> None:
        with patch("core.llm.ollama_provider.LLM", MagicMock()):
            provider = OllamaProvider()
        provider.validate_request(LLMRequest(prompt="hello"))  # should not raise

    def test_empty_prompt_raises(self) -> None:
        with patch("core.llm.ollama_provider.LLM", MagicMock()):
            provider = OllamaProvider()
        with pytest.raises(ValueError):
            provider.validate_request(LLMRequest(prompt=""))


# ---------------------------------------------------------------------------
# Test agenerate (inherited)
# ---------------------------------------------------------------------------
class TestAgenerate:
    def test_agenerate_delegates_to_generate(self) -> None:
        """The BaseLLM default agenerate runs generate in a thread."""
        with patch("core.llm.ollama_provider.LLM", MagicMock()):
            provider = OllamaProvider()
        provider._llm = MagicMock()
        provider._llm.generate = MagicMock(return_value="async-hello")
        import asyncio
        resp = asyncio.run(provider.agenerate(LLMRequest(prompt="hi")))
        assert resp.text == "async-hello"
