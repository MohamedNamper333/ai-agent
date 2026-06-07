"""Ollama provider — wraps the existing core.model.LLM class.

This provider is the local/free default for SIMPLE-level requests.
It reuses the existing LLM implementation (which already handles
GPT4All fallback and Ollama HTTP transport).
"""
from __future__ import annotations

import time
from typing import Any, Iterator, Optional

import requests

from core.model import LLM

from .base import (
    BaseLLM,
    LLMError,
    LLMRequest,
    LLMResponse,
    ProviderUnavailable,
    ReasoningLevel,
)


class OllamaProvider(BaseLLM):
    """Provider for local Ollama models. Wraps core.model.LLM."""

    provider_name = "ollama"

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        url: str = "http://localhost:11434",
        **kwargs: Any,
    ):
        super().__init__(model, **kwargs)
        self.url = url.rstrip("/")
        self._llm: Any = None
        self._init_llm()

    def _init_llm(self) -> None:
        try:
            self._llm = LLM(model_ref=self.model)
        except Exception as e:
            raise ProviderUnavailable(
                f"Failed to initialize LLM: {e}",
                provider=self.provider_name,
            )

    def is_available(self) -> bool:
        if self._llm is None:
            return False
        try:
            r = requests.get(f"{self.url}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def validate_request(self, request: LLMRequest) -> None:
        if not request.prompt or not request.prompt.strip():
            raise ValueError("prompt cannot be empty")

    def generate(self, request: LLMRequest) -> LLMResponse:
        if self._llm is None:
            raise ProviderUnavailable(
                "Ollama LLM not initialized", provider=self.provider_name
            )
        self._record_call()
        start = time.time()
        try:
            text = self._llm.generate(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stop=request.stop,
                stream=False,
                retries=1,
            )
        except TypeError:
            try:
                text = self._llm.generate(
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    stop=request.stop,
                    stream=False,
                )
            except Exception as e:
                self._record_error()
                raise LLMError(
                    f"Ollama generate failed: {e}",
                    provider=self.provider_name,
                    level=request.level,
                ) from e
        except Exception as e:
            self._record_error()
            raise LLMError(
                f"Ollama generate failed: {e}",
                provider=self.provider_name,
                level=request.level,
            ) from e

        latency_ms = (time.time() - start) * 1000
        return LLMResponse(
            text=text or "",
            model=self.model,
            provider=self.provider_name,
            level=request.level,
            latency_ms=latency_ms,
        )

    def stream(self, request: LLMRequest) -> Iterator[str]:
        if self._llm is None:
            raise ProviderUnavailable(
                "Ollama LLM not initialized", provider=self.provider_name
            )
        self._record_call()
        try:
            try:
                gen = self._llm.generate(
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    stop=request.stop,
                    stream=True,
                    retries=1,
                )
            except TypeError:
                gen = self._llm.generate(
                    prompt=request.prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    stop=request.stop,
                    stream=True,
                )
        except Exception as e:
            self._record_error()
            raise LLMError(
                f"Ollama stream failed: {e}",
                provider=self.provider_name,
                level=request.level,
            ) from e
        try:
            for chunk in gen:
                if chunk:
                    yield chunk
        except Exception as e:
            self._record_error()
            raise LLMError(
                f"Ollama stream iteration failed: {e}",
                provider=self.provider_name,
                level=request.level,
            ) from e

    def get_stats(self) -> dict[str, Any]:
        base = super().get_stats()
        base["url"] = self.url
        base["available"] = self.is_available()
        return base
