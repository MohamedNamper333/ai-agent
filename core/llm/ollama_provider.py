"""Ollama provider that wraps the existing ``core.model.LLM``.

Why wrap instead of reimplement?
* The legacy ``LLM`` class already handles retries, timeouts, streaming,
  GPT4All fallback, and the HTTP layer. Reimplementing would duplicate
  hundreds of lines and risk breaking the 71 existing tests.
* The router only cares about the abstract :class:`BaseLLM` surface, so a
  thin adapter is enough.

The provider is fail-soft: if Ollama is not running we still return a
provider instance whose ``is_available()`` is False. The router handles
fallback.
"""

from __future__ import annotations

import asyncio
import time
from typing import Iterator, Optional, Union

import requests

from core.llm.base import BaseLLM, LLMError, LLMRequest, LLMResponse


class OllamaProvider(BaseLLM):
    """Adapter exposing :class:`core.model.LLM` through the ``BaseLLM`` surface.

    Parameters
    ----------
    url : str
        Ollama base URL. Defaults to ``http://localhost:11434``.
    model : str
        Default model tag (e.g. ``qwen2.5:7b``).
    legacy : LLM | None
        Pre-built legacy ``LLM`` instance. When supplied it is used as-is;
        otherwise a new ``LLM`` is constructed.
    """

    name = "ollama"

    def __init__(
        self,
        url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b",
        legacy=None,
    ) -> None:
        self.url = url
        self.model = model
        self._legacy = legacy
        self._available_cache: Optional[bool] = None
        self._available_checked_at: float = 0.0
        self._availability_ttl = 5.0

    def _get_legacy(self):
        """Lazy-import the legacy LLM so the provider can be constructed
        in test environments that lack Ollama installed locally."""
        if self._legacy is not None:
            return self._legacy
        from core.model import LLM  # local import; avoids hard dependency

        return LLM()

    def _is_available_impl(self) -> bool:
        try:
            import requests
        except Exception:
            return False
        try:
            r = requests.get(f"{self.url.rstrip('/')}/api/tags", timeout=2)
            return bool(r.status_code == 200)
        except Exception:
            return False

    def is_available(self) -> bool:
        if self._available_cache is not None:
            return self._available_cache
        self._available_cache = self._is_available_impl()
        self._available_checked_at = time.monotonic()
        return self._available_cache

    def generate(self, request: LLMRequest) -> Union[str, Iterator[str]]:
        if not self.is_available():
            raise LLMError(
                f"Ollama not reachable at {self.url}",
                provider=self.name,
                retryable=False,
            )
        legacy = self._get_legacy()
        # Map abstract request → legacy kwargs.
        kwargs = {}
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.stop is not None:
            kwargs["stop"] = request.stop

        start = time.monotonic()

        try:
            if request.stream:
                return self._wrap_stream(legacy, request.prompt, kwargs, start)
            text = legacy.generate(request.prompt, stream=False, **kwargs)
        except TypeError as exc:
            # Some legacy ``generate`` signatures don't accept every kwarg.
            # Retry with the bare minimum; providers that truly need a kwarg
            # will surface a real LLMError below.
            try:
                if request.stream:
                    return self._wrap_stream(legacy, request.prompt, {}, start)
                text = legacy.generate(request.prompt, stream=False)
            except Exception as exc2:  # pragma: no cover - defensive
                raise LLMError(
                    f"Ollama generate failed: {exc2}",
                    provider=self.name,
                    retryable=True,
                ) from exc2
        except Exception as exc:  # pragma: no cover - defensive
            raise LLMError(
                f"Ollama generate failed: {exc}",
                provider=self.name,
                retryable=True,
            ) from exc
        latency_ms = int((time.monotonic() - start) * 1000)
        return text

    def _wrap_stream(self, legacy, prompt: str, kwargs: dict, start: float):
        """Wrap a legacy streaming call in a chunks iterator that records
        final latency. We can't return both an iterator and a response object,
        so we rely on the router to read chunks and rebuild the text.
        """
        try:
            gen = legacy.generate(prompt, stream=True, **kwargs)
        except TypeError:
            gen = legacy.generate(prompt, stream=True)
        except Exception as exc:  # pragma: no cover - defensive
            raise LLMError(
                f"Ollama stream failed: {exc}",
                provider=self.name,
                retryable=True,
            ) from exc

        for chunk in gen:
            yield chunk

    async def agenerate(self, request: LLMRequest):  # type: ignore[override]
        # Delegate to legacy async path when present.
        legacy = self._get_legacy()
        agenerate = getattr(legacy, "agenerate", None)
        if agenerate is None:
            return await super().agenerate(request)

        return await asyncio.to_thread(
            agenerate,
            request.prompt,
            request.max_tokens,
        )


__all__ = ["OllamaProvider"]
