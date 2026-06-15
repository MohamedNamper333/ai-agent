"""LLM configuration loaded from environment via config.env_loader.

Two providers are configured:
  - Ollama (local, free, default for SIMPLE level)
  - OpenCode Zen (cloud, free, default for MODERATE / DEEP levels)
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    from config.env_loader import get_env
except ImportError:
    BASE_DIR_FALLBACK = Path(__file__).resolve().parent.parent.parent

    def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
        return os.environ.get(key, default)


# Free models available on OpenCode Zen.
# Users can override any of these via env vars (LLM_SIMPLE_MODEL etc.).
OPENCODE_ZEN_FREE_MODELS = {
    "minimax-m3-free": "opencode/minimax-m3-free",
    "big-pickle": "opencode/big-pickle",
    "deepseek-v4-flash-free": "opencode/deepseek-v4-flash-free",
    "nemotron-3-ultra-free": "opencode/nemotron-3-ultra-free",
    "miimo-v2.5-free": "opencode/miimo-v2.5-free",
}


def _ensure_zen_prefix(model: str) -> str:
    """Ensure an OpenCode Zen model id carries the 'opencode/' prefix.

    Bare short names (e.g. "deepseek-v4-flash-free") stored in
    ``LLMConfig.moderate_model`` / ``deep_model`` are normalised to the
    ``opencode/<model>`` form expected by the provider API. Values that
    are already prefixed (e.g. user override ``opencode/big-pickle``)
    are returned unchanged.
    """
    if model.startswith("opencode/"):
        return model
    return f"opencode/{model}"


@dataclass
class LLMConfig:
    """Configuration for LLM providers.

    All values loaded from env via get_env(). See .env.example for keys.
    """

    # Ollama (local, free)
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_enabled: bool = True

    # OpenCode Zen (cloud, free)
    opencode_zen_url: str = "https://opencode.ai/zen/v1"
    opencode_zen_key: str = ""
    opencode_zen_enabled: bool = True

    # Model routing per level
    simple_model: str = "phi3.5:mini"
    moderate_model: str = "qwen2.5:7b"
    deep_model: str = "deepseek-r1:7b"

    # Behaviour
    auto_route: bool = True
    default_level: str = "simple"
    request_timeout: int = 120
    max_retries: int = 2

    def has_any_provider(self) -> bool:
        """True if at least one provider is enabled + usable."""
        if self.ollama_enabled:
            return True
        if self.opencode_zen_enabled and self.opencode_zen_key:
            return True
        return False

    def resolve_for_level(self, level: str) -> tuple[str, str]:
        """Return (provider_name, model_id) for a given level.

        Provider name is one of: "ollama", "opencode_zen".
        Zen model ids are normalised to the "opencode/<model>" form.
        """
        if level == "deep":
            return ("opencode_zen", _ensure_zen_prefix(self.deep_model))
        if level == "moderate":
            return ("opencode_zen", _ensure_zen_prefix(self.moderate_model))
        if self.ollama_enabled:
            return ("ollama", self.simple_model)
        if self.opencode_zen_enabled and self.opencode_zen_key:
            return ("opencode_zen", _ensure_zen_prefix(self.moderate_model))
        return ("ollama", self.simple_model)


def get_llm_config() -> LLMConfig:
    """Load LLM config from environment (os.environ > .env > defaults)."""
    return LLMConfig(
        ollama_url=get_env("OLLAMA_URL", "http://localhost:11434") or "http://localhost:11434",
        ollama_model=get_env("OLLAMA_MODEL", "qwen3:8b") or "qwen3:8b",
        ollama_enabled=(get_env("OLLAMA_ENABLED", "true") or "true").lower() in ("1", "true", "yes"),
        opencode_zen_url=get_env("OPENCODE_ZEN_URL", "https://opencode.ai/zen/v1")
        or "https://opencode.ai/zen/v1",
        opencode_zen_key=get_env("OPENCODE_ZEN_KEY", "") or "",
        opencode_zen_enabled=(get_env("OPENCODE_ZEN_ENABLED", "true") or "true").lower()
        in ("1", "true", "yes"),
        simple_model=get_env("LLM_SIMPLE_MODEL", "qwen3:8b") or "qwen3:8b",
        moderate_model=get_env("LLM_MODERATE_MODEL", "deepseek-v4-flash-free")
        or "deepseek-v4-flash-free",
        deep_model=get_env("LLM_DEEP_MODEL", "big-pickle") or "big-pickle",
        auto_route=(get_env("LLM_AUTO_ROUTE", "true") or "true").lower() in ("1", "true", "yes"),
        default_level=get_env("LLM_DEFAULT_LEVEL", "simple") or "simple",
        request_timeout=int(get_env("LLM_REQUEST_TIMEOUT", "120") or 120),
        max_retries=int(get_env("LLM_MAX_RETRIES", "2") or 2),
    )


def has_any_provider() -> bool:
    """True if at least one LLM provider is enabled + usable from current env.

    Convenience wrapper around ``LLMConfig.has_any_provider`` that always
    reads the latest env values (mirrors ``get_llm_config``).
    """
    return get_llm_config().has_any_provider()


def resolve_for_level(level: str) -> tuple[str, str]:
    """Return ``(provider_name, model_id)`` for the given level from current env.

    Convenience wrapper around ``LLMConfig.resolve_for_level`` that always
    reads the latest env values.
    """
    return get_llm_config().resolve_for_level(level)
