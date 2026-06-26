"""LLMRouter — auto-routes requests to the best free provider.

Strategy (free-tier only):
  - SIMPLE   -> Ollama (local, no rate limit, low latency)
  - MODERATE -> OpenCode Zen with deepseek-v4-flash-free
  - DEEP     -> OpenCode Zen with big-pickle

If the chosen provider is unavailable or fails, the router automatically
falls back to the other provider.  model_override forces a specific
provider:model pair.

The router exposes the same surface as core.model.LLM.generate/stream/
agenerate so it can be a drop-in replacement for `self.model` in
core/agent.py.
"""
from __future__ import annotations

import logging
from typing import Any, Iterator, Optional, Union

from .base import (
    AllProvidersFailed,
    BaseLLM,
    LLMError,
    LLMRequest,
    LLMResponse,
    ProviderUnavailable,
    ReasoningLevel,
    normalize_level,
)
from .config import LLMConfig, get_llm_config


logger = logging.getLogger(__name__)


# Heuristic signals for complexity classification.
DEEP_KEYWORDS = (
    "analyze", "analysis", "design", "architect", "architectural",
    "compare", "evaluate", "implement", "refactor", "optimize",
    "plan ", "design ", "investigate", "research", "review",
    "comprehensive", "detailed", "in-depth", "thorough",
    "explain why", "step by step", "trade-off", "tradeoff",
    "build a", "create a system", "write a program",
)
MODERATE_KEYWORDS = (
    "summarize", "summary", "list ", "describe", "explain", "what is",
    "how does", "translate", "convert", "find ", "show me", "give me",
    "create", "write", "generate", "draft", "rewrite",
)


class LLMRouter:
    """Routes LLMRequest to the best free provider/model.

    Drop-in replacement for the legacy LLM class — same `generate(...)`
    and `stream(...)` signatures.
    """

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or get_llm_config()
        self._ollama: Optional[BaseLLM] = None
        self._zen: Optional[BaseLLM] = None
        self._stats: dict[str, Any] = {
            "total_requests": 0,
            "by_level": {"simple": 0, "moderate": 0, "deep": 0},
            "by_provider": {"ollama": 0, "opencode_zen": 0},
            "by_model": {},
            "fallbacks": 0,
            "errors": 0,
        }

    @property
    def ollama(self) -> BaseLLM:
        """Ollama."""
        if self._ollama is None and self.config.ollama_enabled:
            from .ollama_provider import OllamaProvider

            self._ollama = OllamaProvider(
                model=self.config.simple_model,
                url=self.config.ollama_url,
            )
        if self._ollama is None:
            raise ProviderUnavailable(
                "Ollama is disabled in config", provider="ollama"
            )
        return self._ollama

    @property
    def zen(self) -> BaseLLM:
        """Zen."""
        if self._zen is None and self.config.opencode_zen_enabled:
            from .opencode_zen_provider import OpenCodeZenProvider

            if not self.config.opencode_zen_key:
                raise ProviderUnavailable(
                    "OPENCODE_ZEN_KEY not configured", provider="opencode_zen"
                )
            self._zen = OpenCodeZenProvider(
                model=self.config.deep_model,
                api_key=self.config.opencode_zen_key,
                base_url=self.config.opencode_zen_url,
            )
        if self._zen is None:
            raise ProviderUnavailable(
                "OpenCode Zen is disabled in config", provider="opencode_zen"
            )
        return self._zen

    def classify(self, prompt: str) -> ReasoningLevel:
        """Heuristic complexity classification.

        Returns DEEP for long, design-style requests; MODERATE for
        multi-step tasks that benefit from stronger reasoning; SIMPLE
        otherwise.
        """
        if not self.config.auto_route:
            return normalize_level(self.config.default_level)

        text = (prompt or "").lower().strip()
        words = text.split()
        word_count = len(words)

        if word_count > 50:
            return ReasoningLevel.DEEP
        if any(kw in text for kw in DEEP_KEYWORDS):
            return ReasoningLevel.DEEP
        if text.count("?") >= 2 or text.count("\n") >= 4:
            return ReasoningLevel.DEEP

        if word_count > 15:
            return ReasoningLevel.MODERATE
        if any(kw in text for kw in MODERATE_KEYWORDS):
            return ReasoningLevel.MODERATE

        return ReasoningLevel.SIMPLE

    def _build_request(
        self,
        prompt: str,
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[list[str]] = None,
        level: Optional[Union[ReasoningLevel, str]] = None,
        model_override: Optional[str] = None,
    ) -> LLMRequest:
        if level is None:
            lvl = self.classify(prompt)
        else:
            lvl = normalize_level(level)
        return LLMRequest(
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            level=lvl,
            model_override=model_override,
        )

    def _pick_provider(self, request: LLMRequest) -> BaseLLM:
        if request.model_override:
            m = request.model_override.lower()
            zen_names = {
                "minimax-m3-free", "big-pickle", "deepseek-v4-flash-free",
                "nemotron-3-ultra-free", "miimo-v2.5-free",
            }
            if m.startswith("opencode/") or m in zen_names:
                return self.zen
            if m.startswith("ollama:"):
                return self.ollama
            if m == self.config.ollama_model.lower():
                return self.ollama

        # OpenCodeZen is PRIMARY for all levels.
        # Ollama (local) is FALLBACK only when OpenCodeZen is unavailable.
        if request.level == ReasoningLevel.SIMPLE:
            try:
                return self.zen   # fast model (deepseek-v4-flash-free)
            except ProviderUnavailable:
                return self.ollama
        # MODERATE / DEEP → always OpenCodeZen (big-pickle or deepseek)
        try:
            return self.zen
        except ProviderUnavailable:
            return self.ollama

    def _fallback_provider(self, failed: BaseLLM) -> Optional[BaseLLM]:
        if failed.provider_name == "ollama":
            try:
                return self.zen
            except ProviderUnavailable:
                return None
        if failed.provider_name == "opencode_zen":
            try:
                return self.ollama
            except ProviderUnavailable:
                return None
        return None

    def _set_provider_model(self, provider: BaseLLM, level: ReasoningLevel) -> None:
        if provider.provider_name == "ollama":
            provider.model = self.config.simple_model
        elif level == ReasoningLevel.MODERATE:
            provider.model = self.config.moderate_model
        elif level == ReasoningLevel.DEEP:
            provider.model = self.config.deep_model

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[list[str]] = None,
        stream: bool = False,
        retries: int = 0,
        system: Optional[str] = None,
        level: Optional[Union[ReasoningLevel, str]] = None,
        model_override: Optional[str] = None,
    ) -> Union[str, Iterator[str]]:
        """Drop-in shim for core.model.LLM.generate()."""
        request = self._build_request(
            prompt=prompt,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            level=level,
            model_override=model_override,
        )
        if stream:
            return self._stream_with_fallback(request)
        response = self.generate_full(request)
        return response.text

    def _stream_with_fallback(self, request: LLMRequest) -> Iterator[str]:
        provider = self._pick_provider(request)
        if request.model_override:
            provider.model = request.model_override
        else:
            self._set_provider_model(provider, request.level)
        try:
            yield from provider.stream(request)
            return
        except LLMError as primary_err:
            self._stats["errors"] += 1
            logger.warning(
                "Provider %s stream failed: %s; trying fallback",
                provider.provider_name, primary_err,
            )
            fb = self._fallback_provider(provider)
            if fb is None:
                raise AllProvidersFailed(
                    f"All providers failed. Primary: {primary_err}",
                    provider=provider.provider_name,
                    level=request.level,
                ) from primary_err
            self._stats["fallbacks"] += 1
            self._set_provider_model(fb, request.level)
            try:
                yield from fb.stream(request)
            except LLMError as fb_err:
                raise AllProvidersFailed(
                    f"All providers failed. Primary: {primary_err}; Fallback: {fb_err}",
                    provider=fb.provider_name,
                    level=request.level,
                ) from fb_err

    def stream(self, prompt: str, **kwargs: Any) -> Iterator[str]:
        """Drop-in shim for core.model.LLM.stream()."""
        request = self._build_request(prompt, **kwargs)
        return self._stream_with_fallback(request)

    async def agenerate(self, prompt: str, **kwargs: Any) -> str:
        """Async wrapper — returns just the response text."""
        import asyncio

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: self.generate(prompt, **kwargs)
        )

    def generate_full(self, request: LLMRequest) -> LLMResponse:
        """Generate a full LLMResponse with metadata. With fallback."""
        self._stats["total_requests"] += 1
        self._stats["by_level"][request.level.value] += 1

        primary = self._pick_provider(request)
        if request.model_override:
            primary.model = request.model_override
        else:
            self._set_provider_model(primary, request.level)

        try:
            response = primary.generate(request)
            self._stats["by_provider"][primary.provider_name] += 1
            self._stats["by_model"][response.model] = (
                self._stats["by_model"].get(response.model, 0) + 1
            )
            return response
        except LLMError as primary_err:
            self._stats["errors"] += 1
            logger.warning(
                "Provider %s failed: %s; trying fallback",
                primary.provider_name, primary_err,
            )
            fb = self._fallback_provider(primary)
            if fb is None:
                raise AllProvidersFailed(
                    f"All providers failed. Primary: {primary_err}",
                    provider=primary.provider_name,
                    level=request.level,
                ) from primary_err
            self._stats["fallbacks"] += 1
            self._set_provider_model(fb, request.level)
            try:
                response = fb.generate(request)
                self._stats["by_provider"][fb.provider_name] += 1
                self._stats["by_model"][response.model] = (
                    self._stats["by_model"].get(response.model, 0) + 1
                )
                return response
            except LLMError as fb_err:
                raise AllProvidersFailed(
                    f"All providers failed. Primary: {primary_err}; Fallback: {fb_err}",
                    provider=fb.provider_name,
                    level=request.level,
                ) from fb_err

    def get_stats(self) -> dict[str, Any]:
        """Return hit rate, miss count, eviction count, and current size."""
        return {
            **self._stats,
            "config": {
                "ollama_url": self.config.ollama_url,
                "ollama_model": self.config.ollama_model,
                "ollama_enabled": self.config.ollama_enabled,
                "zen_url": self.config.opencode_zen_url,
                "zen_configured": bool(self.config.opencode_zen_key),
                "zen_enabled": self.config.opencode_zen_enabled,
                "simple_model": self.config.simple_model,
                "moderate_model": self.config.moderate_model,
                "deep_model": self.config.deep_model,
                "auto_route": self.config.auto_route,
                "default_level": self.config.default_level,
            },
        }

    def reset_stats(self) -> None:
        """Reset stats."""
        self._stats = {
            "total_requests": 0,
            "by_level": {"simple": 0, "moderate": 0, "deep": 0},
            "by_provider": {"ollama": 0, "opencode_zen": 0},
            "by_model": {},
            "fallbacks": 0,
            "errors": 0,
        }
