"""End-to-end integration tests for W1 (Provider + Telemetry + CoT + Agent)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.agent import Agent
from core.llm import LLMError, LLMRequest, LLMResponse, LLMRouter
from core.reasoning import CoTEngine, ReasoningChain
from core.telemetry import Telemetry


class _StubProvider:
    """Generic stub provider that records calls and returns canned text."""

    def __init__(self, name: str, text: str = "ok", available: bool = True):
        self.name = name
        self._text = text
        self._available = available
        self.calls: list = []

    def is_available(self) -> bool:
        return self._available

    def generate(self, request):
        self.calls.append(request)
        return LLMResponse(
            text=self._text,
            model=request.model or "stub-model",
            tokens_used=10,
        )

    def agenerate(self, request):
        return self.generate(request)

    def stream(self, request):
        yield self._text


class _FailingOllama(_StubProvider):
    """Provider that raises a retryable LLMError to trigger fallback."""

    def __init__(self):
        super().__init__("ollama", available=True)

    def generate(self, request):
        self.calls.append(request)
        raise LLMError("simulated ollama failure", retryable=True, provider="ollama")

    def agenerate(self, request):
        return self.generate(request)


class _StubLegacyLLM:
    """Stub for the legacy LLM (self.model) used by Agent.chat fast path."""

    def __init__(self, text: str = "stub-response"):
        self._text = text
        self._ollama_model = "stub-model"
        self._use_ollama = True

    def generate(self, prompt, max_tokens=None, temperature=None, stop=None,
                 stream: bool = False, retries: int = 0):
        if stream:
            raise ValueError(
                "Legacy stub does not support stream=True on generate(); "
                "use .stream() for streaming output."
            )
        return self._text

    def stream(self, prompt, max_tokens=None, **kwargs):
        yield self._text

    async def agenerate(self, prompt, **kwargs):
        return self._text


class _CoTFakeRouter:
    """Fake router compatible with CoTEngine's expectations."""

    def __init__(self, text: str):
        self.text = text
        self.calls: list = []

    def generate(self, request):
        self.calls.append(request)
        return LLMResponse(text=self.text, model="fake-cot", tokens_used=20)

    def agenerate(self, request):
        return self.generate(request)


class _NullCache:
    """Inert cache shim for tests that bypass Agent.__init__ (no RAG / plugins / scheduler).

    Agent.chat() only reads `self._cache.get(key)` after a `config.CACHE_TTL > 0`
    guard. Returning ``None`` here short-circuits the cache branch cleanly so the
    rest of the chat pipeline runs unchanged.
    """

    def get(self, key):
        return None

    def set(self, key, value):
        return None

    def clear(self):
        return None


def _build_agent(router, telemetry, *, with_cot: bool = True):
    """Build an Agent bypassing __init__ to avoid RAG/scheduler/plugin loading."""
    agent = object.__new__(Agent)
    agent.model = _StubLegacyLLM()
    agent.tools = MagicMock()
    agent.memory = MagicMock()
    agent.context = MagicMock()
    agent.llm_router = router
    agent.telemetry = telemetry
    agent._is_simple_query = MagicMock(return_value=True)
    agent._retriever = None
    agent._fast_mode = False
    agent._cache = _NullCache()
    if with_cot:
        agent.cot = CoTEngine(router)
    return agent


def _read_telemetry_events(path: Path) -> list:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").strip().split("\n")
        if line
    ]


class TestW1Integration:
    """Verify W1 components (provider + telemetry + CoT + agent) work together."""

    def test_router_provider_telemetry_chain(self, tmp_path):
        """LLMRouter -> provider -> Telemetry: a call flows through and is recorded."""
        ollama = _StubProvider("ollama", text="hello from ollama")
        router = LLMRouter(ollama=ollama)
        tel = Telemetry(log_dir=str(tmp_path), max_events=100, enabled=True)

        with tel.track("llm_call", model="qwen2.5:7b", level="simple"):
            response = router.generate(LLMRequest(
                prompt="test prompt",
                model="qwen2.5:7b",
                max_tokens=64,
            ))

        assert response.text == "hello from ollama"
        assert len(ollama.calls) == 1
        assert ollama.calls[0].prompt == "test prompt"

        events = _read_telemetry_events(tmp_path / "telemetry.jsonl")
        assert len(events) == 1
        assert events[0]["name"] == "llm_call"
        assert events[0]["status"] == "ok"
        assert events[0]["data"]["model"] == "qwen2.5:7b"

    def test_agent_think_uses_cot_engine(self):
        """Agent.think() uses CoTEngine to produce a ReasoningChain with answer."""
        router = _CoTFakeRouter(text="step 1: think about the problem\nfinal: 42")
        tel = Telemetry(enabled=False)
        agent = _build_agent(router, tel, with_cot=True)

        chain = agent.think("plan the auth system", level="deep")

        assert isinstance(chain, ReasoningChain)
        assert chain.answer == "42"
        assert len(router.calls) == 1

    def test_telemetry_persists_to_jsonl(self, tmp_path):
        """Agent.chat() writes events to telemetry.jsonl."""
        router = MagicMock()
        tel = Telemetry(log_dir=str(tmp_path), max_events=100, enabled=True)
        agent = _build_agent(router, tel, with_cot=False)

        result = agent.chat("hello world")

        assert result is not None
        events = _read_telemetry_events(tmp_path / "telemetry.jsonl")
        assert len(events) >= 1
        chat_events = [e for e in events if e["name"] == "chat"]
        assert len(chat_events) >= 1
        assert chat_events[0]["status"] == "ok"

    def test_fallback_to_secondary_provider(self):
        """When primary fails with retryable error, router falls back to secondary."""
        primary = _FailingOllama()
        secondary = _StubProvider("opencode_zen", text="fallback-via-zen")
        router = LLMRouter(ollama=primary, opencode_zen=secondary)

        response = router.generate(LLMRequest(
            prompt="hello",
            model="qwen2.5:7b",
            max_tokens=64,
        ))

        assert response.text == "fallback-via-zen"
        assert len(primary.calls) == 1
        assert len(secondary.calls) == 1
