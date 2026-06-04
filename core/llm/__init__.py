"""core.llm — provider-agnostic LLM abstraction layer.

Exposes the small surface that ``core/agent.py`` and tests consume.

Public API
----------
BaseLLM, LLMRequest, LLMResponse, ReasoningLevel, LLMError
LLMConfig
OllamaProvider
OpenAICompatProvider, build_opencode_zen
LLMRouter
FREE_MODELS
"""

from core.llm.base import (
    BaseLLM,
    LLMError,
    LLMRequest,
    LLMResponse,
    ReasoningLevel,
)
from core.llm.config import LLMConfig
from core.llm.ollama_provider import OllamaProvider
from core.llm.openai_compat_provider import (
    OpenAICompatProvider,
    build_opencode_zen,
)
from core.llm.router import (
    DEFAULT_MODEL_BY_LEVEL,
    DEEP_KEYWORDS,
    FREE_MODELS,
    LLMRouter,
    LONG_CONTEXT_THRESHOLD,
    MODERATE_KEYWORDS,
)

__all__ = [
    "BaseLLM",
    "LLMConfig",
    "LLMError",
    "LLMRequest",
    "LLMResponse",
    "LLMRouter",
    "OllamaProvider",
    "OpenAICompatProvider",
    "ReasoningLevel",
    "build_opencode_zen",
    "DEFAULT_MODEL_BY_LEVEL",
    "DEEP_KEYWORDS",
    "FREE_MODELS",
    "LONG_CONTEXT_THRESHOLD",
    "MODERATE_KEYWORDS",
]
