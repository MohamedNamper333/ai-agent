"""core/gpu_config.py — GPU Optimization for GTX 1660 Super (6GB VRAM)

Configures Ollama and the agent for maximum performance on:
  - GTX 1660 Super: 6GB VRAM, 1408 CUDA cores, 336 GB/s bandwidth
  - 24GB system RAM

Key constraints:
  - Qwen3:8b Q4_K_M: ~5.2GB VRAM (fits in 6GB)
  - Qwen3:4b Q4_K_M: ~2.8GB VRAM (comfortable headroom)
  - Max batch size: 4-8 (VRAM bound)
  - Expected throughput: 20-40 tokens/sec with full GPU offload
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from typing import Optional


@dataclass
class GPUProfile:
    name: str
    vram_gb: float
    recommended_model: str
    recommended_quantization: str
    num_gpu_layers: int           # Layers to offload to GPU (-1 = all)
    context_size: int             # Optimal context for VRAM
    batch_size: int
    expected_tokens_per_sec: int


# GTX 1660 Super profile
GTX_1660_SUPER = GPUProfile(
    name="GTX 1660 Super",
    vram_gb=6.0,
    recommended_model="qwen3:8b",
    recommended_quantization="Q4_K_M",    # 5.2GB — fits in 6GB with headroom
    num_gpu_layers=-1,                     # Offload ALL layers to GPU
    context_size=8192,                     # Safe for 6GB (32K needs ~8GB)
    batch_size=4,
    expected_tokens_per_sec=25,
)

# Fallback if 8b doesn't fit (e.g., after fragmentation)
GTX_1660_SUPER_SAFE = GPUProfile(
    name="GTX 1660 Super (safe mode)",
    vram_gb=6.0,
    recommended_model="qwen3:4b",
    recommended_quantization="Q4_K_M",    # 2.8GB — comfortable
    num_gpu_layers=-1,
    context_size=16384,
    batch_size=8,
    expected_tokens_per_sec=45,
)


def configure_for_gpu() -> dict:
    """
    Set environment variables to maximize GTX 1660 Super performance.
    Call this at startup before loading any models.
    """
    profile = GTX_1660_SUPER

    env_vars = {
        # Ollama GPU settings
        "OLLAMA_NUM_GPU": str(profile.num_gpu_layers),
        "OLLAMA_GPU_LAYERS": str(profile.num_gpu_layers),
        "OLLAMA_CONTEXT_SIZE": str(profile.context_size),

        # CUDA optimization
        "CUDA_VISIBLE_DEVICES": "0",
        "CUDA_MODULE_LOADING": "LAZY",

        # Memory optimization
        "OLLAMA_MAX_LOADED_MODELS": "1",  # Only 1 model at a time in 6GB
        "OLLAMA_NUM_PARALLEL": "1",       # Sequential — more stable on 6GB

        # Performance
        "OLLAMA_FLASH_ATTENTION": "1",    # Flash attention (if supported)
        "OLLAMA_KV_CACHE_TYPE": "q8_0",  # Compressed KV cache
    }

    for key, value in env_vars.items():
        os.environ[key] = value

    return {
        "profile": profile.name,
        "model": profile.recommended_model,
        "quantization": profile.recommended_quantization,
        "gpu_layers": profile.num_gpu_layers,
        "context_size": profile.context_size,
        "expected_speed": f"~{profile.expected_tokens_per_sec} tokens/sec",
        "vram_usage": "~5.2GB / 6GB",
        "env_vars_set": list(env_vars.keys()),
    }


def check_gpu_status() -> dict:
    """Check current GPU memory usage and temperature."""
    result = {
        "gpu_available": False,
        "vram_total_gb": 0.0,
        "vram_used_gb": 0.0,
        "vram_free_gb": 0.0,
        "temperature_c": None,
        "utilization_pct": None,
    }

    try:
        # nvidia-smi query
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits",
        ], timeout=5).decode().strip()

        parts = [p.strip() for p in out.split(",")]
        if len(parts) >= 3:
            result["gpu_available"] = True
            result["vram_total_gb"] = round(int(parts[0]) / 1024, 2)
            result["vram_used_gb"] = round(int(parts[1]) / 1024, 2)
            result["vram_free_gb"] = round(int(parts[2]) / 1024, 2)
            if len(parts) >= 4 and parts[3].isdigit():
                result["temperature_c"] = int(parts[3])
            if len(parts) >= 5 and parts[4].isdigit():
                result["utilization_pct"] = int(parts[4])
    except Exception:
        pass

    # Ollama GPU detection fallback
    if not result["gpu_available"]:
        try:
            import requests
            r = requests.get("http://localhost:11434/api/tags", timeout=2)
            if r.status_code == 200:
                result["ollama_running"] = True
        except Exception:
            pass

    return result


def get_recommended_model() -> str:
    """
    Return the best model for current VRAM availability.
    Falls back to safer model if VRAM is tight.
    """
    status = check_gpu_status()
    if not status["gpu_available"]:
        return "qwen3:4b"  # Safer without GPU confirmation

    free_vram = status.get("vram_free_gb", 0)
    if free_vram >= 5.0:
        return GTX_1660_SUPER.recommended_model      # qwen3:8b
    elif free_vram >= 2.5:
        return GTX_1660_SUPER_SAFE.recommended_model  # qwen3:4b
    else:
        return "qwen3:1.7b"  # Minimal fallback


def get_ollama_options() -> dict:
    """
    Return Ollama generation options optimized for GTX 1660 Super.
    Pass these as `options` in Ollama API calls.
    """
    return {
        "num_gpu": -1,           # Use all GPU layers
        "num_ctx": 8192,         # Context window (safe for 6GB)
        "num_batch": 256,        # Batch size for prompt processing
        "num_thread": 4,         # CPU threads for non-GPU ops
        "f16_kv": True,          # Half-precision KV cache
        "low_vram": False,       # We have enough VRAM
        "numa": False,
    }
