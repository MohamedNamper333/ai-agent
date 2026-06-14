"""core/llm/model_selector.py — Unified Model Selector

Detects and lists ALL available models across:
  - Ollama (local, any loaded model)
  - GPT4All (local, any .gguf file found)
  - OpenCodeZen (cloud API — fetches live model list)

Usage:
    selector = ModelSelector()
    models = selector.get_all_models()
    selector.switch(model_id="qwen3:8b", provider="ollama")
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class ModelInfo:
    model_id: str           # e.g. "qwen3:8b" or "deepseek-v4-flash-free"
    provider: str           # "ollama" | "gpt4all" | "opencode_zen"
    display_name: str
    context_window: int     # tokens
    is_free: bool
    is_local: bool
    description: str = ""
    size_gb: float = 0.0

    @property
    def label(self) -> str:
        tag = "🖥 Local" if self.is_local else "☁ Cloud"
        free = " (Free)" if self.is_free else ""
        return f"{tag} | {self.display_name}{free}"


class ModelSelector:
    """Detects available models and switches between them."""

    # Known OpenCodeZen models with metadata
    _OCZ_KNOWN: list[dict] = [
        {"id": "deepseek-v4-flash-free",   "name": "DeepSeek V4 Flash",    "ctx": 65536,  "desc": "Fast, strong reasoning"},
        {"id": "big-pickle",               "name": "Big Pickle",            "ctx": 32768,  "desc": "General purpose"},
        {"id": "minimax-m3-free",          "name": "MiniMax M3",            "ctx": 40960,  "desc": "Multilingual, balanced"},
        {"id": "nemotron-3-ultra-free",    "name": "Nemotron 3 Ultra",      "ctx": 32768,  "desc": "NVIDIA, strong STEM"},
        {"id": "miimo-v2.5-free",          "name": "Miimo v2.5",            "ctx": 16384,  "desc": "Lightweight, fast"},
        {"id": "qwen-max-free",            "name": "Qwen Max",              "ctx": 32768,  "desc": "Strong coding + analysis"},
    ]

    def __init__(self):
        self._ocz_key = os.getenv("OPENCODE_ZEN_KEY", "")
        self._ocz_url = os.getenv("OPENCODE_ZEN_URL", "https://opencode.ai/zen/v1")
        self._ollama_url = os.getenv("OLLAMA_BASE", "http://127.0.0.1:11434")

    # ─────────────────────────────────────────
    #  Discovery
    # ─────────────────────────────────────────
    def get_all_models(self) -> list[ModelInfo]:
        """Return all detected models across all providers."""
        models: list[ModelInfo] = []
        models.extend(self._get_ollama_models())
        models.extend(self._get_gpt4all_models())
        models.extend(self._get_opencode_zen_models())
        return models

    def get_models_by_provider(self) -> dict[str, list[ModelInfo]]:
        """Group models by provider."""
        all_m = self.get_all_models()
        result: dict[str, list[ModelInfo]] = {"ollama": [], "gpt4all": [], "opencode_zen": []}
        for m in all_m:
            result.setdefault(m.provider, []).append(m)
        return result

    def _get_ollama_models(self) -> list[ModelInfo]:
        try:
            import requests
            r = requests.get(f"{self._ollama_url}/api/tags", timeout=3)
            if r.status_code != 200:
                return []
            data = r.json()
            models = []
            for m in data.get("models", []):
                name = m.get("name", "")
                size_bytes = m.get("size", 0)
                size_gb = round(size_bytes / 1e9, 1)
                ctx = self._guess_ollama_ctx(name)
                models.append(ModelInfo(
                    model_id=name,
                    provider="ollama",
                    display_name=name,
                    context_window=ctx,
                    is_free=True,
                    is_local=True,
                    size_gb=size_gb,
                    description=f"Local Ollama — {size_gb}GB",
                ))
            return models
        except Exception:
            return []

    def _get_gpt4all_models(self) -> list[ModelInfo]:
        """Scan common GPT4All model directories for .gguf files."""
        search_dirs = [
            Path.home() / ".cache" / "gpt4all",
            Path.home() / ".local" / "share" / "nomic.ai" / "GPT4All",
            Path("models"),
            Path(os.getenv("GPT4ALL_MODEL_DIR", "")),
        ]
        found: list[ModelInfo] = []
        seen: set[str] = set()
        for d in search_dirs:
            if not d.exists():
                continue
            for f in d.glob("*.gguf"):
                if f.name in seen:
                    continue
                seen.add(f.name)
                size_gb = round(f.stat().st_size / 1e9, 1)
                found.append(ModelInfo(
                    model_id=str(f),
                    provider="gpt4all",
                    display_name=f.stem,
                    context_window=4096,
                    is_free=True,
                    is_local=True,
                    size_gb=size_gb,
                    description=f"GPT4All local — {size_gb}GB",
                ))
        return found

    def _get_opencode_zen_models(self) -> list[ModelInfo]:
        """Fetch live model list from OpenCodeZen API, fall back to known list."""
        if not self._ocz_key:
            return []

        # Try live fetch first
        try:
            import requests
            r = requests.get(
                f"{self._ocz_url}/models",
                headers={"Authorization": f"Bearer {self._ocz_key}"},
                timeout=5,
            )
            if r.status_code == 200:
                live_models = r.json().get("data", [])
                if live_models:
                    return [
                        ModelInfo(
                            model_id=m.get("id", ""),
                            provider="opencode_zen",
                            display_name=m.get("id", "").replace("-", " ").title(),
                            context_window=m.get("context_window", 32768),
                            is_free=True,
                            is_local=False,
                            description="OpenCodeZen cloud model",
                        )
                        for m in live_models
                        if m.get("id")
                    ]
        except Exception:
            pass

        # Fall back to known list
        return [
            ModelInfo(
                model_id=m["id"],
                provider="opencode_zen",
                display_name=m["name"],
                context_window=m["ctx"],
                is_free=True,
                is_local=False,
                description=m["desc"],
            )
            for m in self._OCZ_KNOWN
        ]

    # ─────────────────────────────────────────
    #  Switch model in config
    # ─────────────────────────────────────────
    def switch(self, model_id: str, provider: str) -> dict:
        """
        Switch the active model by updating config at runtime.
        Returns status dict.
        """
        import config

        if provider == "ollama":
            config.BACKEND = "ollama"
            config.OLLAMA_MODEL = model_id
            # Update env for sub-processes
            os.environ["BACKEND"] = "ollama"
            os.environ["OLLAMA_MODEL"] = model_id
            return {"status": "ok", "provider": "ollama", "model": model_id}

        elif provider == "gpt4all":
            config.BACKEND = "gpt4all"
            config.MODEL_PATH = model_id
            os.environ["BACKEND"] = "gpt4all"
            os.environ["GPT4ALL_MODEL"] = model_id
            return {"status": "ok", "provider": "gpt4all", "model": model_id}

        elif provider == "opencode_zen":
            config.BACKEND = "opencode_zen"
            # Update the router's moderate/deep model
            try:
                from core.llm.config import get_llm_config
                llm_cfg = get_llm_config()
                llm_cfg.moderate_model = model_id
                llm_cfg.deep_model = model_id
            except Exception:
                pass
            os.environ["BACKEND"] = "opencode_zen"
            os.environ["LLM_MODERATE_MODEL"] = model_id
            os.environ["LLM_DEEP_MODEL"] = model_id
            return {"status": "ok", "provider": "opencode_zen", "model": model_id}

        return {"status": "error", "detail": f"Unknown provider: {provider}"}

    def get_active_model(self) -> dict:
        """Return the currently active model info."""
        import config
        backend = getattr(config, "BACKEND", "ollama")
        if backend == "ollama":
            return {"provider": "ollama", "model": getattr(config, "OLLAMA_MODEL", "qwen3:8b")}
        if backend == "gpt4all":
            return {"provider": "gpt4all", "model": getattr(config, "MODEL_PATH", "")}
        if backend == "opencode_zen":
            return {"provider": "opencode_zen", "model": os.getenv("LLM_MODERATE_MODEL", "big-pickle")}
        return {"provider": backend, "model": "unknown"}

    @staticmethod
    def _guess_ollama_ctx(name: str) -> int:
        n = name.lower()
        if "72b" in n or "70b" in n: return 128000
        if "32b" in n: return 131072
        if "14b" in n or "13b" in n: return 131072
        if "8b" in n or "7b" in n: return 131072
        if "3b" in n: return 32768
        if "1b" in n: return 8192
        if "qwen3" in n: return 131072
        if "llama3" in n: return 128000
        if "mistral" in n: return 32768
        return 32768


# Singleton
_selector: Optional[ModelSelector] = None

def get_model_selector() -> ModelSelector:
    global _selector
    if _selector is None:
        _selector = ModelSelector()
    return _selector
