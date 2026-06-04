"""Tests for core.reasoning.cot — CoTEngine + ReasoningChain + CoTPrompts."""
from __future__ import annotations

import asyncio
import pytest

from core.reasoning import CoTEngine, ReasoningChain
from core.reasoning.cot import MAX_STEPS, CONFIDENCE_THRESHOLD
from core.reasoning.prompts import CoTPrompts
from core.llm.base import LLMResponse


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

class _FakeRouter:
    """Minimal router stub. Records every call; returns canned text."""

    def __init__(self, text: str = "step 1: think about it\nfinal: 42") -> None:
        self.text = text
        self.calls: list[tuple[str, object]] = []

    def generate(self, request):
        self.calls.append(("generate", request))
        return LLMResponse(text=self.text, model="stub", tokens_used=5)

    async def agenerate(self, request):
        self.calls.append(("agenerate", request))
        return LLMResponse(text=self.text, model="stub", tokens_used=5)


class _SyncOnlyRouter:
    """Router without agenerate — exercises the asyncio.to_thread fallback."""

    def __init__(self, text: str = "step 1: t\nfinal: 42") -> None:
        self.text = text
        self.calls: list[object] = []

    def generate(self, request):
        self.calls.append(request)
        return LLMResponse(text=self.text, model="stub", tokens_used=1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def router() -> _FakeRouter:
    return _FakeRouter()


@pytest.fixture
def engine(router: _FakeRouter) -> CoTEngine:
    return CoTEngine(router)


# ---------------------------------------------------------------------------
# ReasoningChain.ok contract
# ---------------------------------------------------------------------------

class TestReasoningChainOk:
    def test_ok_with_answer_and_high_confidence(self):
        chain = ReasoningChain(answer="42", confidence=0.9, steps=[])
        assert chain.ok is True

    def test_ok_empty_answer_is_false(self):
        chain = ReasoningChain(answer="", confidence=0.95, steps=[])
        assert chain.ok is False

    def test_ok_below_threshold_is_false(self):
        chain = ReasoningChain(answer="42", confidence=0.69, steps=[])
        assert chain.ok is False

    def test_ok_at_threshold_is_true(self):
        chain = ReasoningChain(
            answer="42", confidence=CONFIDENCE_THRESHOLD, steps=[]
        )
        assert chain.ok is True


# ---------------------------------------------------------------------------
# CoTEngine.think / think_async
# ---------------------------------------------------------------------------

class TestCoTEngineThink:
    def test_think_returns_reasoning_chain(self, engine, router):
        result = engine.think("What is 6*7?")
        assert isinstance(result, ReasoningChain)
        assert router.calls, "router.generate should have been called"

    def test_think_uses_router_generate_sync(self, engine, router):
        engine.think("q", level="deep")
        assert router.calls[0][0] == "generate"

    def test_think_passes_question_in_prompt(self, engine, router):
        engine.think("unique-question-xyz", level="deep")
        request = router.calls[0][1]
        assert hasattr(request, "prompt")
        assert "unique-question-xyz" in request.prompt

    def test_think_async_prefers_router_agenerate(self, engine, router):
        asyncio.run(engine.think_async("q", level="deep"))
        assert router.calls[0][0] == "agenerate"

    def test_think_async_falls_back_when_no_agenerate(self):
        sync_only = _SyncOnlyRouter()
        eng = CoTEngine(sync_only)
        result = asyncio.run(eng.think_async("q"))
        assert isinstance(result, ReasoningChain)
        assert sync_only.calls, "sync generate must run via to_thread"


# ---------------------------------------------------------------------------
# Confidence estimation
# ---------------------------------------------------------------------------

class TestConfidenceEstimation:
    def test_confidence_low_when_no_final_answer(self):
        router = _FakeRouter(text="step 1: just thinking, no conclusion")
        eng = CoTEngine(router)
        result = eng.think("q")
        assert result.confidence < CONFIDENCE_THRESHOLD

    def test_confidence_penalty_for_uncertain_marker(self):
        # Clean answer — no hedging
        clean = _FakeRouter(text="step 1: math\nfinal: 42")
        clean_result = CoTEngine(clean).think("q")
        # Same steps but hedged with an uncertain marker
        hedged = _FakeRouter(text="step 1: maybe math\nfinal: 42")
        hedged_result = CoTEngine(hedged).think("q")
        assert hedged_result.confidence < clean_result.confidence


# ---------------------------------------------------------------------------
# CoTPrompts helper
# ---------------------------------------------------------------------------

class TestCoTPrompts:
    def test_format_step_helper(self):
        out = CoTPrompts.format_step(3, "consider edge cases")
        assert "3" in out
        assert "consider edge cases" in out
