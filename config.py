import os
from pathlib import Path

BASE_DIR = Path(__file__).parent

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

# Backend
BACKEND = _cfg.get("BACKEND", "auto")

# Ollama
OLLAMA_MODEL = _cfg.get("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_BASE = _cfg.get("OLLAMA_BASE", "http://127.0.0.1:11434")

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

# Resolve GGUF model path
if MODEL_PATH and not os.path.isabs(MODEL_PATH):
    p = BASE_DIR / MODEL_PATH
    if p.exists():
        MODEL_PATH = str(p)

if not MODEL_PATH or not os.path.exists(MODEL_PATH):
    gguf_files = list(BASE_DIR.glob("models/*.gguf"))
    if gguf_files:
        MODEL_PATH = str(gguf_files[0])
