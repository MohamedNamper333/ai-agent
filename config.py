import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).parent

# Add config dir to path for env_loader
sys.path.insert(0, str(BASE_DIR / "config"))
from env_loader import get_env

def _read_config():
    config = {}
    cfg_path = BASE_DIR / "config.txt"
    if cfg_path.exists():
        for line in cfg_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            config[key.strip()] = val.strip()
    return config

_cfg = _read_config()

SECRET_KEY = get_env("SECRET_KEY")
API_KEY_HASH_SALT = get_env("API_KEY_HASH_SALT")
CORS_ORIGINS = get_env("CORS_ORIGINS")
LOG_LEVEL = get_env("LOG_LEVEL")

# Backend
BACKEND = _cfg.get("BACKEND", "auto")

# Ollama
OLLAMA_MODEL = _cfg.get("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_BASE = _cfg.get("OLLAMA_BASE", "http://127.0.0.1:11434")

# GPT4All
GPT4ALL_MODEL = _cfg.get("GPT4ALL_MODEL", "")
GPT4ALL_MODEL_DIR = _cfg.get("GPT4ALL_MODEL_DIR", str(Path.home() / ".cache" / "gpt4all"))

# Direct GGUF (llama-cpp-python)
MODEL_PATH = _cfg.get("MODEL_PATH", "")
N_GPU_LAYERS = int(_cfg.get("N_GPU_LAYERS", "-1"))
N_THREADS = int(_cfg.get("N_THREADS", "6"))

# Common
N_CTX = int(_cfg.get("N_CTX", "8192"))
TEMP = float(_cfg.get("TEMP", "0.7"))
MAX_TOKENS = int(_cfg.get("MAX_TOKENS", "2048"))
SYSTEM_PROMPT = _cfg.get("SYSTEM_PROMPT", "You are a helpful AI assistant.")
WEB_HOST = _cfg.get("WEB_HOST", "127.0.0.1")
WEB_PORT = int(_cfg.get("WEB_PORT", "8080"))
DB_PATH = _cfg.get("DB_PATH", "memory_store.json")

# Performance & intelligence
FAST_MODE = _cfg.get("FAST_MODE", "auto")
CACHE_TTL = int(_cfg.get("CACHE_TTL", "300"))
RAG_ENABLED = _cfg.get("RAG_ENABLED", "true").lower() == "true"

# Tools filter: comma-separated, empty = all enabled
TOOLS_ENABLED = [t.strip() for t in _cfg.get("TOOLS_ENABLED", "").split(",") if t.strip()]

# Resolve GGUF model path
if MODEL_PATH and not os.path.isabs(MODEL_PATH):
    p = BASE_DIR / MODEL_PATH
    if p.exists():
        MODEL_PATH = str(p)

if not MODEL_PATH or not os.path.exists(MODEL_PATH):
    gguf_files = list(BASE_DIR.glob("models/*.gguf"))
    if gguf_files:
        MODEL_PATH = str(gguf_files[0])
