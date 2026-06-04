"""OpenAI-compatible provider.

Handles BOTH OpenAI's own API and OpenCode Zen (which is OpenAI-compatible
but exposed at ``https://opencode.ai/zen/v1``). The provider is configured
purely via ``base_url`` and ``api_key`` — there is no separate "OpenAI vs
OpenCode Zen" code path.

The ``openai`` Python SDK is optional. When it isn't installed, the
provider still instantiates, exposes a stable interface, and ``is_available``
returns False so the router can pick a different provider. Tests can
construct it without the SDK.
"""

from __future__ import annotations

import json
import time
from typing import Any, Iterator, Optional, Union

from core.llm.base import BaseLLM, LLMError, LLMRequest, LLMResponse


def _try_import_openai():
    """Import the openai package lazily. Returns the module or None.

    We deliberately do not raise — the framework runs without it.
    """
    try:
        import openai  # type: ignore
        return openai
    except Exception:
        return None


class OpenAICompatProvider(BaseLLM):
    """OpenAI-compatible chat completions provider.

    Parameters
    ----------
    api_key : str
        API key. Empty string disables the provider (``is_available`` False).
    base_url : str
        API base URL. For OpenCode Zen use ``https://opencode.ai/zen/v1``.
    default_model : str
        Model used when ``LLMRequest.model`` is empty.
    provider_name : str
        Human-readable name (used in error messages and telemetry).
    timeout : float
        Per-request timeout in seconds.
    """

    name = "openai_compat"

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        default_model: str = "gpt-4o-mini",
        provider_name: str = "openai_compat",
        timeout: float = 300.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.provider_name = provider_name
        self.timeout = timeout
        self._client = None
        self._supported_models: Optional[set[str]] = None

    def _get_client(self):
        """Build (or return cached) sync OpenAI client."""
        if self._client is not None:
            return self._client
        openai = _try_import_openai()
        if openai is None or not self.api_key:
            return None
        try:
            self._client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
            )
        except Exception:
            self._client = None
        return self._client

    def _effective_model(self, request: LLMRequest) -> str:
        return request.model or self.default_model

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        if _try_import_openai() is None:
            return False
        return True

    def supports(self, model: str) -> bool:
        """When the user has a known catalog we keep it in ``_supported_models``.

        We populate lazily from a class-level constant the first time
        ``supports`` is called for a non-empty model.
        """
        if not model:
            return True
        if self._supported_models is None:
            return True
        return model in self._supported_models

    def set_supported_models(self, models: list[str]) -> None:
        """Restrict the provider to a specific model catalog. Used by the
        router to scope OpenCode Zen to its 5 free models."""
        self._supported_models = set(models)

    def _build_messages(self, prompt: str) -> list[dict]:
        """Split a concatenated prompt into messages. Heuristic: if the
        prompt contains ``\n\nUser:``/``\n\nAssistant:`` markers, parse
        them. Otherwise treat the whole prompt as a single user message.
        """
        prompt = prompt.strip()
        if not prompt:
            return [{"role": "user", "content": ""}]

        # Naive splitter. Good enough for an MVP; downstream callers can
        # always pass pre-formatted ``messages`` via a future field.
        messages = []
        current = []
        for line in prompt.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("system:"):
                if current:
                    flushed = "\n".join(current).strip()
                    if flushed:
                        messages.append({"role": "user", "content": flushed})
                    current = []
                messages.append(
                    {"role": "system", "content": stripped[len("system:"):].strip()}
                )
            elif stripped.lower().startswith(("user:", "human:")):
                if current:
                    flushed = "\n".join(current).strip()
                    if flushed:
                        messages.append({"role": "user", "content": flushed})
                    current = []
                current.append(stripped.split(":", 1)[1].strip())
            elif stripped.lower().startswith(("assistant:", "ai:")):
                if current:
                    flushed = "\n".join(current).strip()
                    if flushed:
                        messages.append({"role": "user", "content": flushed})
                    current = []
                messages.append(
                    {"role": "assistant", "content": stripped.split(":", 1)[1].strip()}
                )
            else:
                current.append(line)
        if current:
            flushed = "\n".join(current).strip()
            if flushed:
                messages.append({"role": "user", "content": flushed})
        if not messages:
            messages.append({"role": "user", "content": prompt})
        return messages

    def _call_once(self, request: LLMRequest, stream: bool) -> Any:
        client = self._get_client()
        if client is None:
            raise LLMError(
                f"{self.provider_name} unavailable (missing SDK or api key)",
                provider=self.provider_name,
                retryable=False,
            )
        model = self._effective_model(request)
        messages = self._build_messages(request.prompt)
        kwargs: dict = {
            "model": model,
            "messages": messages,
        }
        if request.max_tokens is not None:
            kwargs["max_tokens"] = request.max_tokens
        if request.temperature is not None:
            kwargs["temperature"] = request.temperature
        if request.stop is not None:
            kwargs["stop"] = request.stop
        if stream:
            return client.chat.completions.create(stream=True, **kwargs)
        return client.chat.completions.create(**kwargs)

    def generate(self, request: LLMRequest) -> Union[str, Iterator[str]]:
        if not self.is_available():
            raise LLMError(
                f"{self.provider_name} not configured (missing api_key or SDK)",
                provider=self.provider_name,
                retryable=False,
            )
        start = time.monotonic()
        try:
            if request.stream:
                return self._wrap_stream(request, start)
            response = self._call_once(request, stream=False)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(
                f"{self.provider_name} generate failed: {exc}",
                provider=self.provider_name,
                retryable=True,
            ) from exc

        try:
            text = response.choices[0].message.content or ""
            model = getattr(response, "model", self._effective_model(request))
        except Exception:
            text = ""
            model = self._effective_model(request)

        latency_ms = int((time.monotonic() - start) * 1000)
        return text

    def _wrap_stream(self, request: LLMRequest, start: float) -> Iterator[str]:
        response = self._call_once(request, stream=True)
        try:
            for chunk in response:
                try:
                    delta = chunk.choices[0].delta
                    piece = getattr(delta, "content", None) or ""
                except Exception:
                    piece = ""
                if piece:
                    yield piece
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(
                f"{self.provider_name} stream failed: {exc}",
                provider=self.provider_name,
                retryable=True,
            ) from exc

    async def agenerate(self, request: LLMRequest):  # type: ignore[override]
        openai = _try_import_openai()
        if openai is None or not self.api_key:
            return await super().agenerate(request)
        if self._client is None:
            try:
                self._client = openai.OpenAI(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    timeout=self.timeout,
                )
            except Exception:
                return await super().agenerate(request)
        return await super().agenerate(request)


def build_opencode_zen(api_key: str, base_url: str, default_model: str) -> OpenAICompatProvider:
    """Factory that returns an OpenAICompatProvider scoped to OpenCode Zen."""
    provider = OpenAICompatProvider(
        api_key=api_key,
        base_url=base_url,
        default_model=default_model,
        provider_name="opencode_zen",
    )
    provider.set_supported_models(
        [
            "minimax-m3-free",
            "big-pickle",
            "deepseek-v4-flash-free",
            "nemotron-3-ultra-free",
            "mimo-v2.5-free",
        ]
    )
    return provider


__all__ = ["OpenAICompatProvider", "build_opencode_zen"]
