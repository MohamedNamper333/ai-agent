"""Tests for core.llm.base — ReasoningLevel, dataclasses, exceptions, BaseLLM ABC."""
from __future__ import annotations

import asyncio
from typing import Any, Iterator

import pytest

from core.llm.base import (
    AllProvidersFailed,
    BaseLLM,
    LLMError,
    LLMRequest,
    LLMResponse,
    ProviderUnavailable,
    ReasoningLevel,
    normalize_level,
)


# ---------------------------------------------------------------------------
# ReasoningLevel
# ---------------------------------------------------------------------------
class TestReasoningLevel:
    def test_members(self) -> None:
        assert ReasoningLevel.SIMPLE.value == "simple"
        assert ReasoningLevel.MODERATE.value == "moderate"
        assert ReasoningLevel.DEEP.value == "deep"

    def test_string_enum_comparison(self) -> None:
        # str-Enum: behaves like a string
        assert ReasoningLevel.SIMPLE == "simple"
        assert ReasoningLevel.DEEP == "deep"

    def test_iteration_yields_all_three(self) -> None:
        members = list(ReasoningLevel)
        assert len(members) == 3
        assert ReasoningLevel.SIMPLE in members
        assert ReasoningLevel.MODERATE in members
        assert ReasoningLevel.DEEP in members


# ---------------------------------------------------------------------------
# LLMRequest
# ---------------------------------------------------------------------------
class TestLLMRequest:
    def test_minimal_construction(self) -> None:
        req = LLMRequest(prompt="hello")
        assert req.prompt == "hello"
        assert req.system is None
        assert req.max_tokens is None
        assert req.temperature is None
        assert req.stop is None
        assert req.level is ReasoningLevel.SIMPLE
        assert req.model_override is None

    def test_full_construction(self) -> None:
        req = LLMRequest(
            prompt="p",
            system="you are helpful",
            max_tokens=256,
            temperature=0.5,
            stop=["\n"],
            level=ReasoningLevel.DEEP,
            model_override="big-pickle",
        )
        assert req.system == "you are helpful"
        assert req.max_tokens == 256
        assert req.temperature == 0.5
        assert req.stop == ["\n"]
        assert req.level is ReasoningLevel.DEEP
        assert req.model_override == "big-pickle"

    def test_stop_must_be_list_or_none(self) -> None:
        # No runtime enforcement — dataclass only — but verify the default is None
        req = LLMRequest(prompt="x")
        assert req.stop is None


# ---------------------------------------------------------------------------
# LLMResponse
# ---------------------------------------------------------------------------
class TestLLMResponse:
    def test_total_tokens_sums(self) -> None:
        resp = LLMResponse(
            text="hi", model="m", provider="p", level=ReasoningLevel.SIMPLE,
            input_tokens=10, output_tokens=5,
        )
        assert resp.total_tokens == 15

    def test_total_tokens_default_zero(self) -> None:
        resp = LLMResponse(text="x", model="m", provider="p", level=ReasoningLevel.MODERATE)
        assert resp.total_tokens == 0

    def test_metadata_default_empty_dict(self) -> None:
        resp = LLMResponse(text="x", model="m", provider="p", level=ReasoningLevel.SIMPLE)
        assert resp.metadata == {}
        # Mutating the default must not leak across instances
        resp.metadata["k"] = "v"
        resp2 = LLMResponse(text="x", model="m", provider="p", level=ReasoningLevel.SIMPLE)
        assert resp2.metadata == {}

    def test_to_dict_shape(self) -> None:
        resp = LLMResponse(
            text="hello", model="big-pickle", provider="opencode_zen",
            level=ReasoningLevel.DEEP, input_tokens=3, output_tokens=4,
            latency_ms=12.5, metadata={"raw": "data"},
        )
        d = resp.to_dict()
        assert d == {
            "text": "hello",
            "model": "big-pickle",
            "provider": "opencode_zen",
            "level": "deep",
            "input_tokens": 3,
            "output_tokens": 4,
            "total_tokens": 7,
            "latency_ms": 12.5,
            "metadata": {"raw": "data"},
        }

    def test_to_dict_level_serialized_as_string(self) -> None:
        resp = LLMResponse(text="t", model="m", provider="p", level=ReasoningLevel.MODERATE)
        d = resp.to_dict()
        assert d["level"] == "moderate"
        assert isinstance(d["level"], str)


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------
class TestExceptions:
    def test_llm_error_minimal(self) -> None:
        err = LLMError("boom")
        assert str(err) == "boom"
        assert err.provider == ""
        assert err.level is None
        assert isinstance(err, Exception)

    def test_llm_error_with_provider_and_level(self) -> None:
        err = LLMError("bad", provider="ollama", level=ReasoningLevel.DEEP)
        assert err.provider == "ollama"
        assert err.level is ReasoningLevel.DEEP

    def test_provider_unavailable_is_llm_error(self) -> None:
        err = ProviderUnavailable("down", provider="zen")
        assert isinstance(err, LLMError)
        assert err.provider == "zen"

    def test_all_providers_failed_is_llm_error(self) -> None:
        err = AllProvidersFailed("nothing worked")
        assert isinstance(err, LLMError)
        assert str(err) == "nothing worked"


# ---------------------------------------------------------------------------
# normalize_level
# ---------------------------------------------------------------------------
class TestNormalizeLevel:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (ReasoningLevel.SIMPLE, ReasoningLevel.SIMPLE),
            ("simple", ReasoningLevel.SIMPLE),
            ("SIMPLE", ReasoningLevel.SIMPLE),
            ("Simple", ReasoningLevel.SIMPLE),
            ("moderate", ReasoningLevel.MODERATE),
            ("deep", ReasoningLevel.DEEP),
        ],
    )
    def test_valid_inputs(self, raw: Any, expected: ReasoningLevel) -> None:
        assert normalize_level(raw) is expected

    def test_none_returns_simple(self) -> None:
        assert normalize_level(None) is ReasoningLevel.SIMPLE

    def test_unknown_string_returns_simple(self) -> None:
        # Conservative: unknown level defaults to SIMPLE (not an exception)
        assert normalize_level("ultra-deep") is ReasoningLevel.SIMPLE
        assert normalize_level("") is ReasoningLevel.SIMPLE
        assert normalize_level("garbage") is ReasoningLevel.SIMPLE


# ---------------------------------------------------------------------------
# BaseLLM — concrete subclass needed because it's abstract
# ---------------------------------------------------------------------------
class _StubLLM(BaseLLM):
    """Minimal concrete subclass used to exercise BaseLLM defaults."""

    provider_name = "stub"

    def __init__(self, model: str = "stub-model", **kwargs: Any) -> None:
        super().__init__(model, **kwargs)
        self.generate_calls: list[LLMRequest] = []
        self.available: bool = True
        self.next_response: LLMResponse | None = None

    def is_available(self) -> bool:
        return self.available

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.generate_calls.append(request)
        if self.next_response is not None:
            return self.next_response
        return LLMResponse(
            text="stub-text",
            model=self.model,
            provider=self.provider_name,
            level=request.level,
        )


class TestBaseLLMInit:
    def test_init_stores_model_and_config(self) -> None:
        llm = _StubLLM(model="x", temperature=0.5, top_p=0.9)
        assert llm.model == "x"
        assert llm.config == {"temperature": 0.5, "top_p": 0.9}

    def test_init_zero_counters(self) -> None:
        llm = _StubLLM()
        assert llm._call_count == 0
        assert llm._error_count == 0

    def test_cannot_instantiate_base_llm_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseLLM(model="x")  # type: ignore[abstract]


class TestBaseLLMGenerate:
    def test_generate_returns_response(self) -> None:
        llm = _StubLLM()
        req = LLMRequest(prompt="hi", level=ReasoningLevel.MODERATE)
        resp = llm.generate(req)
        assert resp.text == "stub-text"
        assert resp.provider == "stub"
        assert resp.level is ReasoningLevel.MODERATE
        assert req in llm.generate_calls


class TestBaseLLMStream:
    def test_stream_yields_chunks_of_response(self) -> None:
        llm = _StubLLM()
        llm.next_response = LLMResponse(
            text="abcdefghijklmnopqrstuvwxyz",
            model="m", provider="p", level=ReasoningLevel.SIMPLE,
        )
        req = LLMRequest(prompt="p")
        chunks = list(llm.stream(req))
        joined = "".join(chunks)
        assert joined == "abcdefghijklmnopqrstuvwxyz"
        # Default chunk size is 20
        assert all(len(c) <= 20 for c in chunks)
        assert len(chunks) >= 2  # 26 chars / 20 = 2 chunks

    def test_stream_empty_text_yields_nothing(self) -> None:
        llm = _StubLLM()
        llm.next_response = LLMResponse(
            text="", model="m", provider="p", level=ReasoningLevel.SIMPLE,
        )
        chunks = list(llm.stream(LLMRequest(prompt="p")))
        assert chunks == []


class TestBaseLLMAGenerate:
    def test_agenerate_runs_sync_in_executor(self) -> None:
        llm = _StubLLM()
        req = LLMRequest(prompt="async hi", level=ReasoningLevel.DEEP)
        resp = asyncio.run(llm.agenerate(req))
        assert resp.text == "stub-text"
        assert resp.level is ReasoningLevel.DEEP
        assert req in llm.generate_calls


class TestBaseLLMStats:
    def test_initial_stats(self) -> None:
        llm = _StubLLM(model="alpha")
        stats = llm.get_stats()
        assert stats == {
            "provider": "stub",
            "model": "alpha",
            "call_count": 0,
            "error_count": 0,
        }

    def test_record_call_and_error(self) -> None:
        llm = _StubLLM()
        llm._record_call()
        llm._record_call()
        llm._record_error()
        assert llm.get_stats()["call_count"] == 2
        assert llm.get_stats()["error_count"] == 1

    def test_stats_reflect_current_model(self) -> None:
        llm = _StubLLM(model="beta")
        assert llm.get_stats()["model"] == "beta"


class TestBaseLLMIsAvailable:
    def test_default_stub_returns_true(self) -> None:
        assert _StubLLM().is_available() is True

    def test_toggle(self) -> None:
        llm = _StubLLM()
        llm.available = False
        assert llm.is_available() is False
