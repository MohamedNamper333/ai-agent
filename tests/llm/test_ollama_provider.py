"""Tests for core.llm.ollama_provider.OllamaProvider.

These tests inject a mock legacy (core.model.LLM) to bypass the lazy
import path. HTTP probes and asyncio.to_thread are monkey-patched.
"""
import asyncio
from unittest.mock import MagicMock

import pytest

from core.llm.base import LLMError, LLMRequest
from core.llm.ollama_provider import OllamaProvider


def _build_provider(legacy=None, url="http://localhost:11434"):
    """Construct an OllamaProvider with an injected mock legacy."""
    return OllamaProvider(url=url, model="qwen2.5:7b", legacy=legacy)


# --- is_available -----------------------------------------------------------

def test_is_available_returns_true_on_http_200(monkeypatch):
    fake_response = MagicMock(status_code=200)
    monkeypatch.setattr(
        "core.llm.ollama_provider.requests.get",
        lambda url, timeout: fake_response,
    )
    provider = _build_provider(legacy=MagicMock())
    assert provider.is_available() is True


def test_is_available_returns_false_on_http_non_200(monkeypatch):
    fake_response = MagicMock(status_code=503)
    monkeypatch.setattr(
        "core.llm.ollama_provider.requests.get",
        lambda url, timeout: fake_response,
    )
    provider = _build_provider(legacy=MagicMock())
    assert provider.is_available() is False


def test_is_available_caches_within_ttl(monkeypatch):
    call_count = {"n": 0}

    def fake_get(url, timeout):
        call_count["n"] += 1
        return MagicMock(status_code=200)

    monkeypatch.setattr("core.llm.ollama_provider.requests.get", fake_get)
    # Pin time so the TTL never expires between calls.
    monkeypatch.setattr(
        "core.llm.ollama_provider.time.monotonic",
        lambda: 0.0,
    )
    provider = _build_provider(legacy=MagicMock())
    for _ in range(5):
        assert provider.is_available() is True
    # Conservative upper bound — constructor may probe once, so <= 2.
    assert call_count["n"] <= 2


# --- generate ---------------------------------------------------------------

def test_generate_delegates_to_legacy_with_kwargs():
    legacy = MagicMock()
    legacy.generate.return_value = "hello-from-legacy"

    provider = _build_provider(legacy=legacy)
    # Pre-set the cache so is_available() short-circuits.
    provider._available_cache = True
    provider._available_checked_at = 0.0

    request = LLMRequest(
        prompt="hi",
        max_tokens=50,
        temperature=0.3,
        stop=["END"],
    )
    response = provider.generate(request)

    legacy.generate.assert_called_once()
    call = legacy.generate.call_args
    # Prompt should appear in the first positional arg or as a kwarg.
    if call.args:
        assert call.args[0] == "hi"
    else:
        assert call.kwargs.get("prompt") == "hi"
    # Response may be an LLMResponse or a raw string; both are acceptable.
    text = getattr(response, "text", response)
    assert text == "hello-from-legacy"


def test_generate_raises_llm_error_when_unavailable():
    legacy = MagicMock()
    provider = _build_provider(legacy=legacy)
    # Pre-set cache to False to skip HTTP probe.
    provider._available_cache = False
    provider._available_checked_at = 0.0

    request = LLMRequest(prompt="hi", max_tokens=20)
    with pytest.raises(LLMError) as excinfo:
        provider.generate(request)

    assert excinfo.value.retryable is False
    assert excinfo.value.provider == "ollama"
    legacy.generate.assert_not_called()


# --- agenerate --------------------------------------------------------------

def test_agenerate_uses_asyncio_to_thread(monkeypatch):
    legacy = MagicMock()
    legacy.agenerate = MagicMock(return_value="async-ok")

    provider = _build_provider(legacy=legacy)
    provider._available_cache = True
    provider._available_checked_at = 0.0

    captured = {}

    async def fake_to_thread(callable_, *args):
        captured["callable"] = callable_
        captured["args"] = args
        return callable_(*args)

    monkeypatch.setattr(
        "core.llm.ollama_provider.asyncio.to_thread",
        fake_to_thread,
    )

    request = LLMRequest(prompt="hi", max_tokens=20)
    response = asyncio.run(provider.agenerate(request))

    assert captured["callable"] == legacy.agenerate
    assert captured["args"] == ("hi", 20)
    text = getattr(response, "text", response)
    assert text == "async-ok"
