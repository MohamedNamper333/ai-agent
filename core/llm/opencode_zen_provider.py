"""OpenCode Zen provider — free OpenAI-compatible cloud models.

OpenCode Zen exposes an OpenAI-compatible API at https://opencode.ai/zen/v1
with several free models: minimax-m3-free, big-pickle, deepseek-v4-flash-free,
nemotron-3-ultra-free, miimo-v2.5-free.
"""
from __future__ import annotations

import time
from typing import Any, Iterator, Optional

from .base import (
    BaseLLM,
    LLMError,
    LLMRequest,
    LLMResponse,
    ProviderUnavailable,
    ReasoningLevel,
)


class OpenCodeZenProvider(BaseLLM):
    """Provider for OpenCode Zen — free, OpenAI-compatible API."""

    provider_name = "opencode_zen"

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://opencode.ai/zen/v1",
        **kwargs: Any,
    ):
        super().__init__(model, **kwargs)
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._client: Any = None
        self._init_failed = False

    def _get_client(self) -> Any:
        if self._init_failed:
            raise ProviderUnavailable(
                "OpenCode Zen client previously failed to initialize",
                provider=self.provider_name,
            )
        if self._client is not None:
            return self._client
        if not self.api_key:
            raise ProviderUnavailable(
                "OPENCODE_ZEN_KEY not set in environment",
                provider=self.provider_name,
            )
        try:
            import openai
        except ImportError as e:
            self._init_failed = True
            raise ProviderUnavailable(
                "openai package not installed. Run: pip install openai",
                provider=self.provider_name,
            ) from e
        try:
            self._client = openai.OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
        except Exception as e:
            self._init_failed = True
            raise ProviderUnavailable(
                f"Failed to construct OpenCode Zen client: {e}",
                provider=self.provider_name,
            ) from e
        return self._client

    def is_available(self) -> bool:
        """Return True if Docker is installed and the daemon is running."""
        if not self.api_key:
            return False
        try:
            client = self._get_client()
            client.models.list()
            return True
        except Exception:
            return False

    def _resolve_model_id(self) -> str:
        """Pass model name as-is; user can prefix with 'opencode/' via env."""
        return self.model

    def _build_messages(self, request: LLMRequest) -> list[dict[str, str]]:
        msgs: list[dict[str, str]] = []
        if request.system:
            msgs.append({"role": "system", "content": request.system})
        msgs.append({"role": "user", "content": request.prompt})
        return msgs

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate."""
        client = self._get_client()
        model_id = self._resolve_model_id()
        self._record_call()
        start = time.time()
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=self._build_messages(request),
                max_tokens=request.max_tokens or 2000,
                temperature=0.7 if request.temperature is None else request.temperature,
                stop=request.stop,
                stream=False,
            )
        except Exception as e:
            self._record_error()
            raise LLMError(
                f"OpenCode Zen generate failed: {e}",
                provider=self.provider_name,
                level=request.level,
            ) from e

        latency_ms = (time.time() - start) * 1000
        try:
            choice = resp.choices[0]
            text = choice.message.content or ""
        except (AttributeError, IndexError, KeyError) as e:
            self._record_error()
            raise LLMError(
                f"OpenCode Zen returned unexpected payload: {e}",
                provider=self.provider_name,
                level=request.level,
            ) from e

        usage = getattr(resp, "usage", None)
        input_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "completion_tokens", 0) if usage else 0

        return LLMResponse(
            text=text,
            model=model_id,
            provider=self.provider_name,
            level=request.level,
            input_tokens=int(input_tokens or 0),
            output_tokens=int(output_tokens or 0),
            latency_ms=latency_ms,
            metadata={"raw": resp},
        )

    def stream(self, request: LLMRequest) -> Iterator[str]:
        """Stream."""
        client = self._get_client()
        model_id = self._resolve_model_id()
        self._record_call()
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=self._build_messages(request),
                max_tokens=request.max_tokens or 2000,
                temperature=0.7 if request.temperature is None else request.temperature,
                stop=request.stop,
                stream=True,
            )
        except Exception as e:
            self._record_error()
            raise LLMError(
                f"OpenCode Zen stream failed: {e}",
                provider=self.provider_name,
                level=request.level,
            ) from e

        try:
            for chunk in resp:
                try:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                except (AttributeError, IndexError):
                    continue
        except Exception as e:
            self._record_error()
            raise LLMError(
                f"OpenCode Zen stream iteration failed: {e}",
                provider=self.provider_name,
                level=request.level,
            ) from e

    def get_stats(self) -> dict[str, Any]:
        """Return hit rate, miss count, eviction count, and current size."""
        base = super().get_stats()
        base["base_url"] = self.base_url
        base["available"] = self.is_available()
        base["key_configured"] = bool(self.api_key)
        return base
