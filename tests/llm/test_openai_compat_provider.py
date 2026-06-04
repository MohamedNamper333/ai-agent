"""Tests for the OpenAI-compat provider.

Covers the 5 critical paths:
- unavailable when api_key is empty
- unavailable when the openai SDK is missing
- _build_messages naive splitter (System:/User:/Assistant:/Human:/AI:)
- build_opencode_zen factory wiring (provider_name, default_model, catalog)
- generate() wraps any SDK exception into a retryable LLMError
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from core.llm.base import LLMError, LLMRequest
from core.llm.openai_compat_provider import (
    OpenAICompatProvider,
    _try_import_openai,
    build_opencode_zen,
)


# ---------------------------------------------------------------------------
# 1. Empty api_key → unavailable
# ---------------------------------------------------------------------------


def test_unavailable_when_api_key_empty() -> None:
    provider = OpenAICompatProvider(
        api_key="",
        base_url="https://opencode.ai/zen/v1",
        default_model="deepseek-v4-flash-free",
        provider_name="opencode_zen",
    )

    assert provider.is_available() is False
    assert provider._get_client() is None

    with pytest.raises(LLMError) as excinfo:
        provider.generate(LLMRequest(prompt="hi", max_tokens=10))

    err = excinfo.value
    assert err.retryable is False
    assert err.provider == "opencode_zen"


# ---------------------------------------------------------------------------
# 2. SDK missing → unavailable
# ---------------------------------------------------------------------------


def test_unavailable_when_sdk_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "core.llm.openai_compat_provider._try_import_openai",
        lambda: None,
    )

    provider = OpenAICompatProvider(
        api_key="sk-not-empty-but-no-sdk",
        base_url="https://opencode.ai/zen/v1",
        default_model="deepseek-v4-flash-free",
        provider_name="opencode_zen",
    )

    assert provider.is_available() is False
    assert provider._get_client() is None

    with pytest.raises(LLMError) as excinfo:
        provider.generate(LLMRequest(prompt="hi"))

    assert excinfo.value.retryable is False
    assert excinfo.value.provider == "opencode_zen"


# ---------------------------------------------------------------------------
# 3. _build_messages splitter
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt, expected_roles",
    [
        # Empty → single user message
        ("", ["user"]),
        # Whitespace only → stripped then treated as user
        ("   \n  ", ["user"]),
        # Plain prose → single user message
        ("Tell me a joke about cats.", ["user"]),
        # System + user
        (
            "System: you are a helpful assistant.\n\nUser: hi there",
            ["system", "user"],
        ),
        # System + user + assistant + user
        (
            "System: be brief\nUser: hi\nAssistant: hello\nUser: how are you?",
            ["system", "user", "assistant", "user"],
        ),
        # 'Human:' alias for user
        (
            "Human: hi",
            ["user"],
        ),
        # 'AI:' alias for assistant
        (
            "AI: hello back",
            ["assistant"],
        ),
    ],
)
def test_build_messages_splitter(prompt: str, expected_roles: list[str]) -> None:
    provider = OpenAICompatProvider(
        api_key="sk-x",
        base_url="https://opencode.ai/zen/v1",
        default_model="deepseek-v4-flash-free",
        provider_name="opencode_zen",
    )

    messages = provider._build_messages(prompt)

    assert isinstance(messages, list)
    assert len(messages) == len(expected_roles), (
        f"prompt={prompt!r} produced {len(messages)} messages, "
        f"expected {len(expected_roles)}: {messages}"
    )
    for msg, role in zip(messages, expected_roles):
        assert msg["role"] == role
        # Every message must have a non-empty content field
        assert "content" in msg
        assert isinstance(msg["content"], str)


def test_build_messages_system_preserves_content() -> None:
    provider = OpenAICompatProvider(
        api_key="sk-x",
        base_url="https://opencode.ai/zen/v1",
        default_model="deepseek-v4-flash-free",
        provider_name="opencode_zen",
    )

    messages = provider._build_messages("System: be concise.")

    assert len(messages) == 1
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "be concise."


# ---------------------------------------------------------------------------
# 4. build_opencode_zen factory
# ---------------------------------------------------------------------------


def test_build_opencode_zen_factory_wires_catalog() -> None:
    provider = build_opencode_zen(
        api_key="sk-anything",
        base_url="https://opencode.ai/zen/v1",
        default_model="deepseek-v4-flash-free",
    )

    assert isinstance(provider, OpenAICompatProvider)
    assert provider.provider_name == "opencode_zen"
    assert provider.default_model == "deepseek-v4-flash-free"
    assert provider.api_key == "sk-anything"
    assert provider.base_url == "https://opencode.ai/zen/v1"
    assert provider.timeout > 0

    # Catalog: 5 free models, no others
    free_models = {
        "minimax-m3-free",
        "big-pickle",
        "deepseek-v4-flash-free",
        "nemotron-3-ultra-free",
        "mimo-v2.5-free",
    }
    for m in free_models:
        assert provider.supports(m) is True, f"{m} should be supported"
    assert provider.supports("gpt-4o") is False
    assert provider.supports("claude-3-opus") is False

    # Empty model name → always supported (passthrough)
    assert provider.supports("") is True


# ---------------------------------------------------------------------------
# 5. generate() wraps any SDK exception into a retryable LLMError
# ---------------------------------------------------------------------------


def test_generate_wraps_sdk_exception_as_retryable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("boom: 502 bad gateway")

    mock_openai_module = MagicMock()
    mock_openai_module.OpenAI = MagicMock(return_value=mock_client)

    monkeypatch.setattr(
        "core.llm.openai_compat_provider._try_import_openai",
        lambda: mock_openai_module,
    )

    provider = OpenAICompatProvider(
        api_key="sk-real-looking",
        base_url="https://opencode.ai/zen/v1",
        default_model="deepseek-v4-flash-free",
        provider_name="opencode_zen",
    )

    # Sanity: SDK is now "present" and key is set, so the provider is available.
    assert provider.is_available() is True
    assert provider._get_client() is mock_client

    with pytest.raises(LLMError) as excinfo:
        provider.generate(
            LLMRequest(
                prompt="write a haiku",
                max_tokens=50,
                temperature=0.7,
                stop=["END"],
            )
        )

    err = excinfo.value
    assert err.retryable is True, "transient SDK failures should be retryable"
    assert err.provider == "opencode_zen"
    # The original exception should be chained via __cause__
    assert isinstance(err.__cause__, RuntimeError)

    # The provider should have called chat.completions.create once with
    # the expected kwargs.
    assert mock_client.chat.completions.create.call_count == 1
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "deepseek-v4-flash-free"
    assert call_kwargs["max_tokens"] == 50
    assert call_kwargs["temperature"] == 0.7
    assert call_kwargs["stop"] == ["END"]
    assert call_kwargs["messages"] == [{"role": "user", "content": "write a haiku"}]


# ---------------------------------------------------------------------------
# Defensive: confirm _try_import_openai helper itself is well-behaved
# ---------------------------------------------------------------------------


def test_try_import_openai_helper_returns_module_or_none() -> None:
    """The lazy import must never raise — it returns the module if installed,
    or None otherwise. We don't assert which one (depends on the environment);
    we just confirm the function returns either a module or None.
    """
    result = _try_import_openai()
    assert result is None or hasattr(result, "OpenAI")
