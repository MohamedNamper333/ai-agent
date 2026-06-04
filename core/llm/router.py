"""LLM router.

Picks the right provider/model for a given request. The router is the
single object the agent talks to; it owns one :class:`OllamaProvider` and
one :class:`OpenAICompatProvider` (configured for OpenCode Zen) and exposes
``generate`` / ``agenerate`` that match the legacy ``core.model.LLM``
interface.

Routing signals
---------------
The router inspects a request for several cheap signals:

* ``metadata`` keys (``level`` / ``reasoning_level``) — explicit override.
* Prompt keywords that flag complexity (e.g. "plan", "prove", "analyze").
* Context size > :data:`LONG_CONTEXT_THRESHOLD` tokens.
* Conversation history depth.
* Tool complexity (number of registered tools).

Each signal contributes a score. The cumulative score maps to a
:data:`ReasoningLevel` which in turn picks a model.

The router is fail-soft: when a provider raises :class:`LLMError`, it
walks the fallback list. If everything fails the last error is re-raised.
"""

from __future__ import annotations

import re
import time
from typing import Iterator, Optional, Union

from core.llm.base import (
    BaseLLM,
    LLMError,
    LLMRequest,
    LLMResponse,
    ReasoningLevel,
)
from core.llm.config import LLMConfig
from core.llm.ollama_provider import OllamaProvider
from core.llm.openai_compat_provider import (
    OpenAICompatProvider,
    build_opencode_zen,
)


# Free-model catalog. The router picks from this set; the OpenCode Zen
# provider refuses unknown models.
FREE_MODELS: tuple[str, ...] = (
    "minimax-m3-free",
    "big-pickle",
    "deepseek-v4-flash-free",
    "nemotron-3-ultra-free",
    "mimo-v2.5-free",
)


# Routing tables — these are the public surface for "which model handles
# which level". Update in lockstep with ``FREE_MODELS`` above.
DEFAULT_MODEL_BY_LEVEL: dict[ReasoningLevel, str] = {
    ReasoningLevel.SIMPLE: "qwen2.5:7b",
    ReasoningLevel.MODERATE: "mimo-v2.5-free",
    ReasoningLevel.DEEP: "deepseek-v4-flash-free",
}


# Keyword sets used to score prompt complexity. Kept small and obvious so
# anyone can read the router and understand why a query is "deep".
DEEP_KEYWORDS: frozenset[str] = frozenset(
    {
        "plan", "prove", "verify", "analyze", "design",
        "architecture", "optimize", "refactor", "debug", "compare",
        "evaluate", "synthesize", "reasoning", "theorem", "derive",
        "complex", "in depth", "step by step",
    }
)
MODERATE_KEYWORDS: frozenset[str] = frozenset(
    {
        "write", "generate", "create", "build", "implement",
        "code", "function", "class", "explain", "summarize",
        "translate", "convert", "edit", "fix",
    }
)


LONG_CONTEXT_THRESHOLD = 8000  # characters; cheap proxy for token count


class LLMRouter:
    """Single entry-point for agent / web layer.

    Construct once at process start, then call :meth:`generate` /
    :meth:`agenerate` per request.

    Parameters
    ----------
    config : LLMConfig | None
        Static configuration. When None, defaults are used.
    ollama : OllamaProvider | None
        Pre-built provider. When None, one is built from ``config``.
    opencode_zen : OpenAICompatProvider | None
        Pre-built provider. When None, built only if
        ``config.opencode_zen_api_key`` is set.
    """

    def __init__(
        self,
        config: LLMConfig | None = None,
        ollama: OllamaProvider | None = None,
        opencode_zen: OpenAICompatProvider | None = None,
    ) -> None:
        self.config = config or LLMConfig()
        self.ollama = ollama or OllamaProvider(
            url=self.config.ollama_url,
            model=self.config.ollama_model,
        )
        self.opencode_zen: Optional[OpenAICompatProvider] = opencode_zen
        if self.opencode_zen is None and self.config.opencode_zen_api_key:
            self.opencode_zen = build_opencode_zen(
                api_key=self.config.opencode_zen_api_key,
                base_url=self.config.opencode_zen_base_url,
                default_model=self.config.opencode_zen_default_model,
            )

        # Build the canonical routing table at construction time so it
        # reflects the providers that are actually configured.
        self._model_by_level: dict[ReasoningLevel, str] = dict(
            DEFAULT_MODEL_BY_LEVEL
        )
        # If OpenCode Zen isn't configured, fall back DEEP → Ollama to
        # avoid an instant LLMError.
        if not self._zen_available():
            self._model_by_level[ReasoningLevel.DEEP] = self.config.ollama_model
            self._model_by_level[ReasoningLevel.MODERATE] = self.config.ollama_model

    # -- provider availability ------------------------------------------------

    def _zen_available(self) -> bool:
        return self.opencode_zen is not None and self.opencode_zen.is_available()

    def _ollama_available(self) -> bool:
        try:
            return bool(self.ollama.is_available())
        except Exception:
            return False

    def available_providers(self) -> list[str]:
        names: list[str] = []
        if self._ollama_available():
            names.append("ollama")
        if self._zen_available():
            names.append("opencode_zen")
        return names

    # -- routing logic --------------------------------------------------------

    def _score_prompt(self, prompt: str) -> int:
        text = (prompt or "").lower()
        if not text:
            return 0
        score = 0
        for kw in DEEP_KEYWORDS:
            if re.search(rf"\b{re.escape(kw)}\b", text):
                score += 2
        for kw in MODERATE_KEYWORDS:
            if re.search(rf"\b{re.escape(kw)}\b", text):
                score += 1
        # Long context pushes things deeper even without keywords.
        if len(text) > LONG_CONTEXT_THRESHOLD:
            score += 3
        return score

    def _classify(self, request: LLMRequest) -> ReasoningLevel:
        explicit = request.metadata.get("level") or request.metadata.get(
            "reasoning_level"
        )
        if explicit is not None:
            return ReasoningLevel.from_str(str(explicit))

        if not self.config.auto_route:
            return ReasoningLevel.from_str(self.config.default_level)

        score = self._score_prompt(request.prompt)
        # Conversation history bumps complexity by 1 if > 4 turns.
        history_depth = int(request.metadata.get("history_depth", 0) or 0)
        if history_depth > 4:
            score += 1
        tool_count = int(request.metadata.get("tool_count", 0) or 0)
        if tool_count >= 5:
            score += 1

        return self._score_to_level(score)

    def _score_to_level(self, score: int) -> ReasoningLevel:
        if score >= 4:
            return ReasoningLevel.DEEP
        if score >= 1:
            return ReasoningLevel.MODERATE
        return ReasoningLevel.SIMPLE

    def classify_level(self, prompt: str) -> ReasoningLevel:
        """Public classifier: map a free-form prompt to a :class:`ReasoningLevel`.

        Does NOT touch any provider — it only inspects keywords, prompt
        length, and the configured ``default_level``. Useful for callers
        (UI, telemetry) that want to preview the routing decision before
        issuing a request.
        """
        if not self.config.auto_route:
            return ReasoningLevel.from_str(self.config.default_level)
        score = self._score_prompt(prompt or "")
        return self._score_to_level(score)

    def select_model(self, request: LLMRequest) -> str:
        """Pick the best model for the request.

        Honors an explicit ``request.model`` first, otherwise the level
        table.
        """
        if request.model:
            return request.model
        level = self._classify(request)
        return self._model_by_level[level]

    # -- public generate ------------------------------------------------------

    def _resolve_provider(self, model: str) -> BaseLLM:
        """Pick the provider that owns ``model``.

        Heuristic: anything in the OpenCode Zen free catalog goes to Zen;
        anything else (Ollama tags like ``qwen2.5:7b``) goes to Ollama.
        """
        if model in FREE_MODELS and self._zen_available():
            return self.opencode_zen  # type: ignore[return-value]
        return self.ollama

    def _build_request(self, request: LLMRequest) -> LLMRequest:
        """Mutate a copy of ``request`` so its ``model`` matches what the
        router picked. The caller is not affected."""
        if request.model:
            return request
        chosen = self.select_model(request)
        return LLMRequest(
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=list(request.stop) if request.stop else None,
            stream=request.stream,
            model=chosen,
            metadata=dict(request.metadata),
        )

    def _call_with_fallback(
        self, request: LLMRequest
    ) -> Union[str, Iterator[str]]:
        effective = self._build_request(request)
        chosen_model = effective.model or self.select_model(request)

        primary = self._resolve_provider(chosen_model)
        secondary: Optional[BaseLLM] = None
        if primary is self.ollama and self._zen_available():
            secondary = self.opencode_zen
        elif primary is self.opencode_zen and self._ollama_available():
            secondary = self.ollama

        last_exc: Optional[LLMError] = None
        for provider in (primary, secondary):
            if provider is None:
                continue
            try:
                return provider.generate(effective)
            except LLMError as exc:
                last_exc = exc
                # Retryable → try fallback. Non-retryable → surface now.
                if not exc.retryable:
                    raise
                continue

        raise last_exc or LLMError(
            "No provider available",
            provider="router",
            retryable=False,
        )

    def generate(self, request: LLMRequest) -> Union[str, Iterator[str]]:
        return self._call_with_fallback(request)

    async def agenerate(self, request: LLMRequest) -> Union[str, Iterator[str]]:
        effective = self._build_request(request)
        chosen_model = effective.model or self.select_model(request)
        primary = self._resolve_provider(chosen_model)
        if hasattr(primary, "agenerate"):
            return await primary.agenerate(effective)
        # Fallback to the base async default.
        return await primary.agenerate(effective)

    # -- back-compat with core.model.LLM --------------------------------------

    def generate_text(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[list[str]] = None,
        stream: bool = False,
        retries: int = 0,
    ) -> str:
        """Legacy-friendly shim used by ``core/agent.py``.

        Returns a plain string; for streaming use ``generate`` directly.
        The ``retries`` arg is accepted for API symmetry but the router's
        providers already implement internal retries, so it's a no-op.
        """
        result = self.generate(
            LLMRequest(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=stop,
                stream=False,
            )
        )
        if isinstance(result, LLMResponse):
            return result.text
        if isinstance(result, str):
            return result
        # Provider returned an iterator even though we asked for sync;
        # collapse to text.
        chunks = []
        for chunk in result:
            chunks.append(chunk)
        return "".join(chunks)


__all__ = [
    "LLMRouter",
    "FREE_MODELS",
    "DEFAULT_MODEL_BY_LEVEL",
    "DEEP_KEYWORDS",
    "MODERATE_KEYWORDS",
    "LONG_CONTEXT_THRESHOLD",
]
