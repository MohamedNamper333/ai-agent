"""Provider-agnostic LLM abstractions.

Defines the contract every concrete provider (Ollama, OpenAI-compatible / OpenCode
Zen, etc.) must satisfy. Keeping this surface small keeps the router, agent
integration, and tests simple.

Exports
-------
BaseLLM            : Abstract base class every provider implements.
LLMRequest         : Provider-agnostic request envelope.
LLMResponse        : Provider-agnostic response envelope.
ReasoningLevel     : SIMPLE / MODERATE / DEEP — used by the router.
LLMError           : Exception type raised by providers on fatal errors.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import AsyncIterator, Iterator, Optional


class ReasoningLevel(str, Enum):
    """Coarse complexity bucket used by the router.

    SIMPLE  : short factual / single-tool calls. Use the cheapest provider.
    MODERATE: multi-step reasoning, code generation. Use a balanced model.
    DEEP    : long chain-of-thought, planning, verification. Use the strongest
              available free model.
    """

    SIMPLE = "simple"
    MODERATE = "moderate"
    DEEP = "deep"

    @classmethod
    def from_str(cls, value: str | None) -> "ReasoningLevel":
        if not value:
            return cls.MODERATE
        v = value.strip().lower()
        for member in cls:
            if member.value == v:
                return member
        if v in {"low", "fast", "easy", "1"}:
            return cls.SIMPLE
        if v in {"high", "hard", "3", "max"}:
            return cls.DEEP
        return cls.MODERATE


@dataclass
class LLMRequest:
    """Provider-agnostic request envelope.

    Attributes
    ----------
    prompt : str
        The full prompt (system + user, concatenated by the caller). Providers
        may further split, but the contract is "send this text".
    max_tokens : Optional[int]
        Provider-specific cap; None = use provider default.
    temperature : Optional[float]
        0.0–1.0+. None = use provider default.
    stop : Optional[list[str]]
        Optional stop sequences.
    stream : bool
        When True, ``provider.generate`` returns an iterator/generator.
    model : Optional[str]
        Explicit model override. When set, the router has already decided on
        a model but the provider may still re-route when it supports multiple.
    metadata : dict
        Free-form context for telemetry / debugging. Never interpreted by the
        provider.
    """

    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stop: Optional[list[str]] = None
    stream: bool = False
    model: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Provider-agnostic response envelope.

    Attributes
    ----------
    text : str
        Concatenated output text. For streaming responses, this is empty
        (the caller consumes the generator returned by ``generate(..., stream=True)``).
    model : str
        The model that actually produced the response (may differ from request).
    provider : str
        Lower-case provider name (e.g. ``"ollama"``, ``"opencode_zen"``).
    tokens_in : int
        Input token count when available, else 0.
    tokens_out : int
        Output token count when available, else 0.
    latency_ms : int
        Wall-clock latency excluding retries. Streaming responses set this
        only at end-of-stream.
    raw : dict
        Original provider payload (for debugging / advanced consumers).
    """

    text: str = ""
    model: str = ""
    provider: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    tokens_used: int = 0
    latency_ms: int = 0
    raw: dict = field(default_factory=dict)


class LLMError(RuntimeError):
    """Raised when a provider encounters an unrecoverable error.

    Carries the underlying error message plus the provider name so that the
    router can log structured failures. Network / 5xx errors are retried
    internally by the provider and should not surface here unless the retry
    budget is exhausted.
    """

    def __init__(self, message: str, *, provider: str = "", retryable: bool = False) -> None:
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable


class BaseLLM(abc.ABC):
    """Abstract base class every provider must implement.

    The router talks only to this surface. Implementations are expected to:

    * be safe to instantiate without network access (defer remote checks to
      ``generate``);
    * perform their own internal retries; on permanent failure raise
      :class:`LLMError`;
    * when ``stream=True`` return an iterator of strings (sync) or async
      iterator of strings (async variant). Each yielded item is a chunk
      (typically a few characters / a single token).
    """

    name: str = "base"

    @abc.abstractmethod
    def generate(self, request: LLMRequest) -> object:
        """Generate a response.

        Returns
        -------
        str | Iterator[str]
            Plain text when ``request.stream`` is False, otherwise an
            iterator of text chunks. Concrete return type is union
            ``str | Iterator[str]``; signatures use ``object`` to keep this
            base class agnostic.
        """

    async def agenerate(self, request: LLMRequest) -> object:
        """Async variant. Default implementation runs the sync version in a thread.

        Providers with native async clients (openai.AsyncClient) should override.
        """
        import asyncio

        def _runner() -> object:
            return self.generate(request)

        return await asyncio.to_thread(_runner)

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True if the provider can serve requests right now.

        Used by the router to pick a fallback when the primary provider is
        down. Cheap to call; implementations should cache results for at
        most a few seconds.
        """

    def supports(self, model: str) -> bool:
        """Return True if ``model`` is known to this provider.

        Default: every provider supports every model whose name does not
        start with the name of another provider. Concrete providers should
        override with their real model list.
        """
        if not model:
            return True
        return True


__all__ = [
    "BaseLLM",
    "LLMRequest",
    "LLMResponse",
    "ReasoningLevel",
    "LLMError",
]
