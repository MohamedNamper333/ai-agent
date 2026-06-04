"""Tests for core.llm.router.LLMRouter."""
from __future__ import annotations

from unittest.mock import MagicMock, Mock

import pytest

from core.llm import (
    DEEP_KEYWORDS,
    DEFAULT_MODEL_BY_LEVEL,
    FREE_MODELS,
    LONG_CONTEXT_THRESHOLD,
    LLMError,
    LLMRouter,
    LLMResponse,
    MODERATE_KEYWORDS,
    ReasoningLevel,
)


@pytest.fixture
def router():
    """Build an LLMRouter with both providers mocked and available."""
    ollama = Mock()
    ollama.is_available = Mock(return_value=True)
    ollama.name = "ollama"
    ollama.generate = Mock(
        return_value=LLMResponse(text="default", model="qwen2.5:7b", tokens_used=1)
    )

    zen = Mock()
    zen.is_available = Mock(return_value=True)
    zen.name = "opencode_zen"
    zen.generate = Mock(
        return_value=LLMResponse(
            text="default", model="deepseek-v4-flash-free", tokens_used=1
        )
    )

    return LLMRouter(config=MagicMock(), ollama=ollama, opencode_zen=zen), ollama, zen


def test_free_models_catalog_is_tuple_with_expected_entries():
    assert isinstance(FREE_MODELS, (tuple, list))
    assert len(FREE_MODELS) >= 3
    for model in FREE_MODELS:
        assert isinstance(model, str)
        assert model
    assert "deepseek-v4-flash-free" in FREE_MODELS
    assert "mimo-v2.5-free" in FREE_MODELS


def test_default_model_mapping_covers_all_three_reasoning_levels():
    assert isinstance(DEFAULT_MODEL_BY_LEVEL, dict)
    for level in ReasoningLevel:
        assert level in DEFAULT_MODEL_BY_LEVEL
        assert isinstance(DEFAULT_MODEL_BY_LEVEL[level], str)
        assert DEFAULT_MODEL_BY_LEVEL[level]


def test_keyword_lists_and_context_threshold_are_sane():
    assert isinstance(DEEP_KEYWORDS, (list, tuple, set, frozenset))
    assert isinstance(MODERATE_KEYWORDS, (list, tuple, set, frozenset))
    assert len(DEEP_KEYWORDS) >= 1
    assert len(MODERATE_KEYWORDS) >= 1
    for kw in DEEP_KEYWORDS:
        assert isinstance(kw, str) and kw
    for kw in MODERATE_KEYWORDS:
        assert isinstance(kw, str) and kw
    assert isinstance(LONG_CONTEXT_THRESHOLD, int)
    assert LONG_CONTEXT_THRESHOLD > 0


def test_generate_text_shim_returns_string_and_builds_request(router):
    r, ollama, _zen = router
    ollama.generate = Mock(
        return_value=LLMResponse(text="hello back", model="qwen2.5:7b", tokens_used=5)
    )

    result = r.generate_text("hi", max_tokens=50, temperature=0.3)

    assert isinstance(result, str)
    assert result == "hello back"
    assert ollama.generate.call_count == 1

    call_args = ollama.generate.call_args
    request = call_args[0][0] if call_args[0] else call_args.kwargs.get("request")
    assert request is not None
    assert getattr(request, "prompt", None) == "hi"
    assert getattr(request, "max_tokens", None) == 50
    assert getattr(request, "temperature", None) == 0.3


def test_generate_text_falls_back_to_secondary_on_retryable_error(router):
    r, ollama, zen = router
    ollama.generate = Mock(side_effect=LLMError("network blip", retryable=True))
    zen.generate = Mock(
        return_value=LLMResponse(
            text="from secondary", model="deepseek-v4-flash-free", tokens_used=3
        )
    )

    result = r.generate_text("hello", max_tokens=10)

    assert result == "from secondary"
    assert ollama.generate.call_count == 1
    assert zen.generate.call_count == 1


def test_generate_text_does_not_fallback_on_non_retryable_error(router):
    r, ollama, zen = router
    ollama.generate = Mock(side_effect=LLMError("bad request", retryable=False))
    zen.generate = Mock(
        return_value=LLMResponse(
            text="should not be called", model="deepseek-v4-flash-free", tokens_used=3
        )
    )

    with pytest.raises(LLMError):
        r.generate_text("hello", max_tokens=10)

    assert ollama.generate.call_count == 1
    assert zen.generate.call_count == 0


def test_classify_level_returns_reasoninglevel_enum(router):
    r, ollama, zen = router

    level = r.classify_level("hello world")

    assert isinstance(level, ReasoningLevel)
    assert level != ReasoningLevel.DEEP
    assert ollama.generate.call_count == 0
    assert zen.generate.call_count == 0


def test_classify_level_with_deep_keywords_promotes_level(router):
    r, ollama, zen = router

    deep_words = list(DEEP_KEYWORDS)[:5]
    assert len(deep_words) >= 1, "DEEP_KEYWORDS should be non-empty"
    query = "Please " + " ".join(deep_words) + " this complex problem"
    level = r.classify_level(query)

    assert level in {ReasoningLevel.DEEP, ReasoningLevel.MODERATE}
