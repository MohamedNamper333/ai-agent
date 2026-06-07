"""Base classes for the LLM provider abstraction.

Defines:
- ReasoningLevel: simple | moderate | deep (drives routing)
- LLMRequest / LLMResponse: provider-agnostic data containers
- BaseLLM: ABC for all providers
- LLMError, ProviderUnavailable, AllProvidersFailed: exception hierarchy
"""
from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterator, Optional, Union


class ReasoningLevel(str, Enum):
    """Reasoning depth required for a request. Drives router decisions."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    DEEP = "deep"


@dataclass
class LLMRequest:
    """Provider-agnostic LLM request.

    Attributes:
        prompt: the user / task text
        system: optional system message
        max_tokens: cap on response size
        temperature: 0.0 deterministic -> 1.0 creative
        stop: optional list of stop sequences
        level: routing level (simple/moderate/deep)
        model_override: explicit model id; bypasses routing
    """

    prompt: str
    system: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stop: Optional[list[str]] = None
    level: ReasoningLevel = ReasoningLevel.SIMPLE
    model_override: Optional[str] = None


@dataclass
class LLMResponse:
    """Provider-agnostic LLM response with metadata."""

    text: str
    model: str
    provider: str
    level: ReasoningLevel
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "model": self.model,
            "provider": self.provider,
            "level": self.level.value,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }


class LLMError(Exception):
    """Base exception for LLM errors."""

    def __init__(
        self,
        message: str,
        provider: str = "",
        level: Optional[ReasoningLevel] = None,
    ):
        super().__init__(message)
        self.provider = provider
        self.level = level


class ProviderUnavailable(LLMError):
    """Provider is not configured or unreachable."""


class AllProvidersFailed(LLMError):
    """All configured providers failed for the request."""


class BaseLLM(ABC):
    """Abstract base class for LLM providers.

    Subclasses must implement is_available() and generate().
    Default implementations of stream() (chunked non-streaming),
    agenerate() (async wrapper), and get_stats() are provided.
    """

    provider_name: str = "base"

    def __init__(self, model: str, **kwargs: Any):
        self.model = model
        self.config = kwargs
        self._call_count = 0
        self._error_count = 0

    @abstractmethod
    def is_available(self) -> bool:
        """Whether this provider is reachable / properly configured."""

    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Synchronous generation. Must return a complete LLMResponse."""

    def stream(self, request: LLMRequest) -> Iterator[str]:
        """Streaming generation. Default: chunked non-streaming output.

        Override for true streaming behavior.
        """
        response = self.generate(request)
        text = response.text or ""
        chunk_size = 20
        for i in range(0, len(text), chunk_size):
            yield text[i : i + chunk_size]

    async def agenerate(self, request: LLMRequest) -> LLMResponse:
        """Async generation. Default runs sync generate in executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.generate, request)

    def get_stats(self) -> dict[str, Any]:
        return {
            "provider": self.provider_name,
            "model": self.model,
            "call_count": self._call_count,
            "error_count": self._error_count,
        }

    def _record_call(self) -> None:
        self._call_count += 1

    def _record_error(self) -> None:
        self._error_count += 1


def normalize_level(level: Union[ReasoningLevel, str, None]) -> ReasoningLevel:
    """Coerce string/None to ReasoningLevel enum."""
    if level is None:
        return ReasoningLevel.SIMPLE
    if isinstance(level, ReasoningLevel):
        return level
    try:
        return ReasoningLevel(level)
    except ValueError:
        return ReasoningLevel.SIMPLE
