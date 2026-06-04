"""LLM configuration loaded from environment variables.

The router and providers all read from :class:`LLMConfig`. We deliberately
keep this class small and stdlib-only so it can be instantiated in tests
without any I/O.

Notes
-----
We read directly from :data:`os.environ` rather than going through
``config.env_loader``. The reason is the project root contains a single
``config.py`` module that shadows the ``config/`` package, which would
make ``from config import env_loader`` raise ``ImportError`` during
collection. LLM env vars are user-supplied at runtime, so we do not need
``.env`` file fallback here.

Environment variables
---------------------
OLLAMA_URL                : Ollama base URL (default ``http://localhost:11434``)
OLLAMA_MODEL              : Default Ollama model tag (default ``qwen2.5:7b``)
OPENCODE_ZEN_API_KEY      : OpenCode Zen API key (no default — empty = disabled)
OPENCODE_ZEN_BASE_URL     : API base (default ``https://opencode.ai/zen/v1``)
OPENCODE_ZEN_DEFAULT_MODEL: Default model on OpenCode Zen
                           (default ``deepseek-v4-flash-free``)
LLM_DEFAULT_LEVEL         : ``simple`` / ``moderate`` / ``deep`` (default ``moderate``)
LLM_AUTO_ROUTE            : ``true`` / ``false`` (default ``true``)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


def _truthy(value: str | None) -> bool:
    if not value:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on", "y", "t"}


def _get(key: str, default: str) -> str:
    """Read a string env var with a default. Never raises."""
    value = os.environ.get(key, default)
    if value is None:
        value = default
    return str(value)


@dataclass
class LLMConfig:
    """Static configuration for the LLM provider layer.

    All fields have safe defaults so the framework runs even when no env
    vars are set. Tests can construct this directly without touching the
    environment.
    """

    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"

    opencode_zen_api_key: str = ""
    opencode_zen_base_url: str = "https://opencode.ai/zen/v1"
    opencode_zen_default_model: str = "deepseek-v4-flash-free"

    default_level: str = "moderate"
    auto_route: bool = True

    # Free-model catalog exposed to users. The router uses an internal map
    # for picking; this list is what gets surfaced in docs / help text.
    available_models: list[str] = field(
        default_factory=lambda: [
            "minimax-m3-free",
            "big-pickle",
            "deepseek-v4-flash-free",
            "nemotron-3-ultra-free",
            "mimo-v2.5-free",
        ]
    )

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Build a config from ``os.environ``.

        Falls back to defaults when keys are missing. Never raises.
        """

        ollama_url = _get("OLLAMA_URL", "http://localhost:11434")
        ollama_model = _get("OLLAMA_MODEL", "qwen2.5:7b")

        api_key = _get("OPENCODE_ZEN_API_KEY", "")
        base_url = _get(
            "OPENCODE_ZEN_BASE_URL", "https://opencode.ai/zen/v1"
        )
        default_zen_model = _get(
            "OPENCODE_ZEN_DEFAULT_MODEL", "deepseek-v4-flash-free"
        )

        default_level = _get("LLM_DEFAULT_LEVEL", "moderate")
        auto_route = _truthy(_get("LLM_AUTO_ROUTE", "true"))

        return cls(
            ollama_url=ollama_url,
            ollama_model=ollama_model,
            opencode_zen_api_key=api_key,
            opencode_zen_base_url=base_url,
            opencode_zen_default_model=default_zen_model,
            default_level=default_level,
            auto_route=auto_route,
        )


__all__ = ["LLMConfig"]
