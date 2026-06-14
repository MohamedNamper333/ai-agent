"""LLM Provider Abstraction Layer.

Routes between free local (Ollama) and free cloud (OpenCode Zen) models.
OpenCode Zen is OpenAI-compatible and free for the configured model tier.
"""
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
from .ollama_provider import OllamaProvider
from .opencode_zen_provider import OpenCodeZenProvider
from .router import LLMRouter

try:
    from .model_selector import ModelSelector, ModelInfo, get_model_selector
except ImportError:
    pass

__all__ = [
    "AllProvidersFailed",
    "BaseLLM",
    "LLMConfig",
    "LLMError",
    "LLMRequest",
    "LLMResponse",
    "LLMRouter",
    "OllamaProvider",
    "OpenCodeZenProvider",
    "ProviderUnavailable",
    "ReasoningLevel",
    "get_llm_config",
    "normalize_level",
]
