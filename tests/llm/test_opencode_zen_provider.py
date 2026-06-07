"""Tests for OpenCodeZenProvider — the cloud (free) LLM provider.

Covers:
- Construction, defaults, trailing-slash stripping.
- Lazy client construction, caching, and failure paths.
- Availability probing.
- Message building and model-id resolution.
- Sync generation: defaults, custom params, error wrapping, missing usage.
- Streaming: delta content, falsy-chunk skipping, error wrapping.
- Stats and inheritance surface.
"""
from __future__ import annotations

import sys
from typing import Any, Iterator
from unittest.mock import MagicMock

import pytest

from core.llm.base import (
    LLMError,
    LLMRequest,
    LLMResponse,
    ProviderUnavailable,
    ReasoningLevel,
)
from core.llm.opencode_zen_provider import OpenCodeZenProvider


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _make_provider(
    model: str = "minimax-m3-free",
    api_key: str = "test-api-key-123",
    base_url: str = "https://opencode.ai/zen/v1",
) -> OpenCodeZenProvider:
    return OpenCodeZenProvider(model=model, api_key=api_key, base_url=base_url)


def _make_provider_with_mock_client(
    **kwargs: Any,
) -> tuple[OpenCodeZenProvider, MagicMock]:
    provider = _make_provider(**kwargs)
    mock_client = MagicMock(name="opencode_zen_client")
    provider._client = mock_client
    return provider, mock_client


def _make_chat_response(
    text: str = "Hello from Zen",
    prompt_tokens: int = 5,
    completion_tokens: int = 3,
    model: str = "opencode/minimax-m3-free",
) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = text
    response.usage = MagicMock()
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    response.model = model
    return response


def _make_stream_chunks(texts: list[str | None]) -> Iterator[MagicMock]:
    for t in texts:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = t
        yield chunk


# ------------------------------------------------------------------
# TestInit
# ------------------------------------------------------------------
class TestInit:
    """Provider construction, defaults, and trailing-slash stripping."""

    def test_default_base_url_is_zen_endpoint(self) -> None:
        provider = _make_provider()
        assert provider.base_url == "https://opencode.ai/zen/v1"

    def test_strips_trailing_slash_from_custom_base_url(self) -> None:
        provider = _make_provider(base_url="https://opencode.ai/zen/v1/")
        assert provider.base_url == "https://opencode.ai/zen/v1"

    def test_keeps_custom_base_url_unchanged_when_no_slash(self) -> None:
        provider = _make_provider(base_url="https://custom.zen.example.com")
        assert provider.base_url == "https://custom.zen.example.com"

    def test_required_arguments_stored(self) -> None:
        provider = _make_provider(model="big-pickle", api_key="sk-abc")
        assert provider.model == "big-pickle"
        assert provider.api_key == "sk-abc"

    def test_provider_name_is_opencode_zen(self) -> None:
        provider = _make_provider()
        assert provider.provider_name == "opencode_zen"

    def test_init_client_is_none_init_failed_false(self) -> None:
        provider = _make_provider()
        assert provider._client is None
        assert provider._init_failed is False


# ------------------------------------------------------------------
# TestGetClient
# ------------------------------------------------------------------
class TestGetClient:
    """Lazy client construction, caching, and failure paths."""

    def test_returns_cached_client_on_second_call(self) -> None:
        provider = _make_provider()
        cached = MagicMock(name="cached")
        provider._client = cached
        assert provider._get_client() is cached
        assert provider._get_client() is cached

    def test_short_circuits_when_previously_failed(self) -> None:
        provider = _make_provider()
        provider._init_failed = True
        with pytest.raises(ProviderUnavailable) as exc_info:
            provider._get_client()
        assert "previously failed to initialize" in str(exc_info.value)

    def test_missing_api_key_raises_provider_unavailable(self) -> None:
        provider = _make_provider(api_key="")
        with pytest.raises(ProviderUnavailable) as exc_info:
            provider._get_client()
        assert "OPENCODE_ZEN_KEY" in str(exc_info.value)
        # Missing-key path does NOT set _init_failed (user can set the env var)
        assert provider._init_failed is False

    def test_import_error_sets_init_failed_and_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Force `import openai` to fail by setting sys.modules entry to None
        monkeypatch.setitem(sys.modules, "openai", None)
        provider = _make_provider()
        with pytest.raises(ProviderUnavailable) as exc_info:
            provider._get_client()
        assert "openai package not installed" in str(exc_info.value)
        assert provider._init_failed is True

    def test_openai_constructor_failure_sets_init_failed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_openai = MagicMock()
        fake_openai.OpenAI.side_effect = RuntimeError("network unreachable")
        monkeypatch.setitem(sys.modules, "openai", fake_openai)

        provider = _make_provider()
        with pytest.raises(ProviderUnavailable) as exc_info:
            provider._get_client()
        assert "Failed to construct OpenCode Zen client" in str(exc_info.value)
        assert "network unreachable" in str(exc_info.value)
        assert provider._init_failed is True

    def test_constructs_client_with_base_url_and_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_client_instance = MagicMock(name="client_instance")
        fake_openai = MagicMock()
        fake_openai.OpenAI = MagicMock(return_value=fake_client_instance)
        monkeypatch.setitem(sys.modules, "openai", fake_openai)

        provider = _make_provider(api_key="sk-test-key")
        client = provider._get_client()
        assert client is fake_client_instance
        fake_openai.OpenAI.assert_called_once_with(
            base_url="https://opencode.ai/zen/v1",
            api_key="sk-test-key",
        )
        # Second call returns the cached client
        assert provider._get_client() is fake_client_instance


# ------------------------------------------------------------------
# TestIsAvailable
# ------------------------------------------------------------------
class TestIsAvailable:
    """Availability probes (api_key present + models.list() succeeds)."""

    def test_returns_false_when_no_api_key(self) -> None:
        provider = _make_provider(api_key="")
        assert provider.is_available() is False

    def test_returns_true_when_models_list_succeeds(self) -> None:
        provider, client = _make_provider_with_mock_client()
        client.models.list.return_value = MagicMock()
        assert provider.is_available() is True
        client.models.list.assert_called_once()

    def test_returns_false_when_models_list_raises(self) -> None:
        provider, client = _make_provider_with_mock_client()
        client.models.list.side_effect = RuntimeError("api down")
        assert provider.is_available() is False


# ------------------------------------------------------------------
# TestBuildMessages
# ------------------------------------------------------------------
class TestBuildMessages:
    """Message-list construction (system prompt + user prompt)."""

    def test_build_messages_without_system(self) -> None:
        provider = _make_provider()
        request = LLMRequest(prompt="hi")
        messages = provider._build_messages(request)
        assert messages == [{"role": "user", "content": "hi"}]

    def test_build_messages_with_system(self) -> None:
        provider = _make_provider()
        request = LLMRequest(prompt="hi", system="you are helpful")
        messages = provider._build_messages(request)
        assert messages == [
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hi"},
        ]


# ------------------------------------------------------------------
# TestResolveModelId
# ------------------------------------------------------------------
class TestResolveModelId:
    """Model-id resolution (pass-through — model is already fully qualified)."""

    def test_resolve_model_id_returns_model_unchanged(self) -> None:
        provider = _make_provider(model="opencode/deepseek-v4-flash-free")
        assert provider._resolve_model_id() == "opencode/deepseek-v4-flash-free"


# ------------------------------------------------------------------
# TestGenerate
# ------------------------------------------------------------------
class TestGenerate:
    """Sync generation: LLMResponse shape, defaults, error wrapping, missing usage."""

    def test_generate_returns_llm_response_on_success(self) -> None:
        provider, client = _make_provider_with_mock_client()
        mock_response = _make_chat_response(
            text="hi there", prompt_tokens=7, completion_tokens=2
        )
        client.chat.completions.create.return_value = mock_response
        request = LLMRequest(prompt="hello", level=ReasoningLevel.MODERATE)

        response = provider.generate(request)

        assert isinstance(response, LLMResponse)
        assert response.text == "hi there"
        assert response.provider == "opencode_zen"
        assert response.level == ReasoningLevel.MODERATE
        assert response.input_tokens == 7
        assert response.output_tokens == 2
        assert response.latency_ms >= 0
        assert "raw" in response.metadata
        assert response.metadata["raw"] is mock_response
        client.chat.completions.create.assert_called_once()

    def test_generate_uses_default_max_tokens_and_temperature(self) -> None:
        provider, client = _make_provider_with_mock_client()
        client.chat.completions.create.return_value = _make_chat_response()
        request = LLMRequest(prompt="hello")  # max_tokens=None, temperature=None

        provider.generate(request)

        _, kwargs = client.chat.completions.create.call_args
        assert kwargs["max_tokens"] == 2000
        assert kwargs["temperature"] == 0.7

    def test_generate_passes_custom_max_tokens_and_temperature(self) -> None:
        provider, client = _make_provider_with_mock_client()
        client.chat.completions.create.return_value = _make_chat_response()
        request = LLMRequest(prompt="hello", max_tokens=512, temperature=0.2)

        provider.generate(request)

        _, kwargs = client.chat.completions.create.call_args
        assert kwargs["max_tokens"] == 512
        assert kwargs["temperature"] == 0.2

    def test_generate_wraps_exception_in_llm_error(self) -> None:
        provider, client = _make_provider_with_mock_client()
        client.chat.completions.create.side_effect = RuntimeError("upstream boom")
        request = LLMRequest(prompt="hello")

        with pytest.raises(LLMError) as exc_info:
            provider.generate(request)
        # The original error message or the wrapper label should be visible
        assert (
            "upstream boom" in str(exc_info.value)
            or "OpenCode Zen" in str(exc_info.value)
        )

    def test_generate_requires_initialized_client(self) -> None:
        provider = _make_provider(api_key="")
        with pytest.raises(ProviderUnavailable):
            provider.generate(LLMRequest(prompt="hi"))

    def test_generate_handles_missing_usage_gracefully(self) -> None:
        provider, client = _make_provider_with_mock_client()
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = "ok"
        response.usage = None  # upstream didn't return token usage
        client.chat.completions.create.return_value = response

        out = provider.generate(LLMRequest(prompt="hi"))
        assert out.text == "ok"
        assert out.input_tokens == 0
        assert out.output_tokens == 0


# ------------------------------------------------------------------
# TestStream
# ------------------------------------------------------------------
class TestStream:
    """Streaming yields delta text, skips falsy chunks, wraps errors."""

    def test_stream_yields_delta_content(self) -> None:
        provider, client = _make_provider_with_mock_client()
        client.chat.completions.create.return_value = _make_stream_chunks(
            ["Hel", "lo ", "world"]
        )
        request = LLMRequest(prompt="hi", max_tokens=10, temperature=0.0)

        chunks = list(provider.stream(request))
        assert chunks == ["Hel", "lo ", "world"]

    def test_stream_skips_falsy_deltas(self) -> None:
        provider, client = _make_provider_with_mock_client()
        client.chat.completions.create.return_value = _make_stream_chunks(
            ["a", "", None, "b", " ", None]
        )
        request = LLMRequest(prompt="hi", max_tokens=10, temperature=0.0)

        chunks = list(provider.stream(request))
        assert chunks == ["a", "b", " "]

    def test_stream_wraps_exception_in_llm_error(self) -> None:
        provider, client = _make_provider_with_mock_client()
        client.chat.completions.create.side_effect = RuntimeError("connection reset")
        request = LLMRequest(prompt="hi")

        with pytest.raises(LLMError):
            list(provider.stream(request))


# ------------------------------------------------------------------
# TestGetStats
# ------------------------------------------------------------------
class TestGetStats:
    """get_stats extends base with provider-specific telemetry."""

    def test_get_stats_includes_base_url_availability_key(self) -> None:
        provider, client = _make_provider_with_mock_client()
        client.models.list.return_value = MagicMock()  # is_available → True

        stats = provider.get_stats()
        assert stats["base_url"] == "https://opencode.ai/zen/v1"
        assert stats["available"] is True
        assert stats["key_configured"] is True
        # Inherited from BaseLLM
        assert stats["provider"] == "opencode_zen"
        assert "model" in stats
        assert stats["model"] == "minimax-m3-free"

    def test_get_stats_reports_key_not_configured(self) -> None:
        provider = _make_provider(api_key="")
        stats = provider.get_stats()
        assert stats["key_configured"] is False
        assert stats["available"] is False


# ------------------------------------------------------------------
# TestNoOverride
# ------------------------------------------------------------------
class TestNoOverride:
    """Provider must NOT override agenerate / validate_request (inherit from BaseLLM)."""

    def test_does_not_override_agenerate(self) -> None:
        assert "agenerate" not in OpenCodeZenProvider.__dict__

    def test_does_not_override_validate_request(self) -> None:
        assert "validate_request" not in OpenCodeZenProvider.__dict__

    def test_is_subclass_of_base_llm(self) -> None:
        from core.llm.base import BaseLLM
        assert issubclass(OpenCodeZenProvider, BaseLLM)
