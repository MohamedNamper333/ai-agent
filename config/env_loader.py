import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DEFAULTS = {
    "SECRET_KEY": "change-me-in-production",
    "API_KEY_HASH_SALT": "change-me-in-production",
    "CORS_ORIGINS": "http://localhost:8080",
    "LOG_LEVEL": "INFO",
}


def _load_dotenv(path: Path | None = None) -> dict[str, str]:
    if path is None:
        path = BASE_DIR / ".env"
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env[key.strip()] = value.strip()
    return env


_env_cache: dict[str, str] | None = None


def _get_all() -> dict[str, str]:
    global _env_cache
    if _env_cache is None:
        _env_cache = {**DEFAULTS, **_load_dotenv()}
    return _env_cache


def get_env(key: str, default: str | None = None) -> str:
    all_env = _get_all()
    value = os.environ.get(key, all_env.get(key))
    if value is not None:
        return value
    if default is not None:
        return default
    raise KeyError(f"Environment variable '{key}' is not set and has no default.")
