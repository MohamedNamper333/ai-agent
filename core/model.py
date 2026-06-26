import json
import sys
import time
import hashlib
from typing import Optional, Generator
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

import config

try:
    import requests
except ImportError:
    requests = None

try:
    from gpt4all import GPT4All
except ImportError:
    GPT4All = None


class ModelErrorType(Enum):
    CONNECTION = "connection"
    TIMEOUT = "timeout"
    OVERLOAD = "overload"
    INVALID_REQUEST = "invalid_request"
    UNKNOWN = "unknown"


@dataclass
class ModelError:
    error_type: ModelErrorType
    message: str
    retryable: bool = True
    status_code: int = 0


class LLM:
    MAX_RETRIES = 3
    RETRY_BACKOFF_BASE = 2
    MAX_BACKOFF = 30
    REQUEST_TIMEOUT = 300

    def __init__(self, backend: str = "auto"):
        self._backend = backend
        self._model = None
        self._model_path = ""
        self._ollama_base = "http://127.0.0.1:11434"
        self._ollama_model = ""
        self._use_ollama = False
        self._use_gpt4all = False
        self._gpt4all_model_name = ""
        self._token_cache: dict[str, int] = {}
        self._retry_stats = {"total": 0, "success": 0, "failed": 0}

    @property
    def is_loaded(self) -> bool:
        """Return True if a model backend is ready to generate text."""
        return self._use_ollama or self._use_gpt4all or self._model is not None

    def load(self, model_path: str = "") -> None:
        """Load data from storage."""
        path = model_path or config.MODEL_PATH

        if self._backend == "gpt4all" or (self._backend == "auto" and self._detect_gpt4all()):
            self._load_gpt4all()
            return

        if self._backend == "ollama" or (self._backend == "auto" and self._detect_ollama()):
            self._setup_ollama(config.OLLAMA_MODEL)
            return

        self._load_llama_cpp(path)

    def _detect_gpt4all(self) -> bool:
        if GPT4All is None:
            return False
        if not config.GPT4ALL_MODEL:
            return False
        model_path = Path(config.GPT4ALL_MODEL_DIR) / config.GPT4ALL_MODEL
        return model_path.exists()

    def _load_gpt4all(self) -> None:
        if GPT4All is None:
            raise ImportError("gpt4all not installed. Run: pip install gpt4all")

        if not config.GPT4ALL_MODEL:
            raise ValueError("GPT4ALL_MODEL not set in config.txt")

        model_path = Path(config.GPT4ALL_MODEL_DIR) / config.GPT4ALL_MODEL
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self._use_gpt4all = True
        self._gpt4all_model_name = config.GPT4ALL_MODEL

        print(f"Loading gpt4all model: {self._gpt4all_model_name}")
        print(f"  Path: {model_path}")
        t0 = time.time()

        self._model = GPT4All(
            model_name=config.GPT4ALL_MODEL,
            model_path=config.GPT4ALL_MODEL_DIR,
            allow_download=False,
            verbose=False,
        )

        print(f"Loaded in {time.time() - t0:.1f}s")

    def _detect_ollama(self) -> bool:
        if requests is None:
            return False
        try:
            r = requests.get(f"{self._ollama_base}/api/tags", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def _setup_ollama(self, model_ref: str = "") -> None:
        self._use_ollama = True
        self._ollama_model = model_ref or "qwen2.5:7b"

        if requests is None:
            raise ImportError("requests module required for Ollama backend")

        try:
            r = requests.get(f"{self._ollama_base}/api/tags", timeout=5)
            if r.status_code != 200:
                raise ConnectionError(f"Ollama not responding at {self._ollama_base}")

            models = r.json().get("models", [])
            model_name = self._ollama_model.split(":")[0]

            installed = any(model_name in m.get("name", "") for m in models)
            if not installed:
                print(f"Model '{self._ollama_model}' not found in Ollama.")
                print(f"Pull it: ollama pull {self._ollama_model}")
                print(f"Or run: ollama run {self._ollama_model}")

            print(f"Ollama backend: {self._ollama_model}")
            print(f"  API: {self._ollama_base}")
        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                "Ollama is not running.\n"
                "1. Download from https://ollama.com\n"
                "2. Install and run Ollama\n"
                "3. Run: ollama pull qwen2.5:7b\n"
                "4. Restart this agent"
            )

    def _load_llama_cpp(self, path: str) -> None:
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "llama-cpp-python not installed and Ollama not detected.\n"
                "Option 1: Install Ollama (recommended for Windows)\n"
                "Option 2: pip install llama-cpp-python\n"
                "Option 3: pip install 'llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121'"
            )

        if not path or not Path(path).exists():
            gguf_files = list(Path("models").glob("*.gguf"))
            if gguf_files:
                path = str(gguf_files[0])
            else:
                raise FileNotFoundError(
                    f"No model found. Place a .gguf file in models/ or install Ollama."
                )

        self._model_path = str(path)
        n_gpu = config.N_GPU_LAYERS

        print(f"Loading GGUF model: {Path(self._model_path).name}")
        print(f"  Context: {config.N_CTX} | GPU layers: {n_gpu}")
        t0 = time.time()

        self._model = Llama(
            model_path=self._model_path,
            n_ctx=config.N_CTX,
            n_threads=config.N_THREADS,
            n_gpu_layers=n_gpu,
            verbose=False,
        )

        print(f"Loaded in {time.time() - t0:.1f}s")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop: Optional[list[str]] = None,
        stream: bool = False,
        retries: int = 0,
    ):
        """Generate."""
        max_retries = retries if retries > 0 else self.MAX_RETRIES

        for attempt in range(max_retries):
            try:
                if self._use_gpt4all:
                    return self._gpt4all_generate(prompt, max_tokens, temperature, stop, stream)
                if self._use_ollama:
                    return self._ollama_generate(prompt, max_tokens, temperature, stop, stream)
                return self._llama_generate(prompt, max_tokens, temperature, stop, stream)

            except Exception as e:
                error = self._classify_error(e)
                self._retry_stats["total"] += 1

                if not error.retryable or attempt >= max_retries - 1:
                    self._retry_stats["failed"] += 1
                    if stream:
                        def error_gen():
                            """Error gen."""
                            yield f"\n[Model error: {error.message}]"
                        return error_gen()
                    return f"[Model error: {error.message}]"

                backoff = min(
                    self.RETRY_BACKOFF_BASE ** attempt,
                    self.MAX_BACKOFF
                )
                print(f"[model] Retry {attempt + 1}/{max_retries} after {backoff}s: {error.message}")
                time.sleep(backoff)

        self._retry_stats["failed"] += 1
        if stream:
            def error_gen():
                """Error gen."""
                yield "[Model error: Max retries exceeded]"
            return error_gen()
        return "[Model error: Max retries exceeded]"

    def _classify_error(self, error: Exception) -> ModelError:
        error_str = str(error).lower()

        if isinstance(error, (ConnectionError, ConnectionRefusedError)):
            return ModelError(ModelErrorType.CONNECTION, str(error), retryable=True)

        if "timeout" in error_str or isinstance(error, TimeoutError):
            return ModelError(ModelErrorType.TIMEOUT, str(error), retryable=True)

        if "503" in error_str or "overloaded" in error_str or "too many" in error_str:
            return ModelError(ModelErrorType.OVERLOAD, str(error), retryable=True)

        if "400" in error_str or "invalid" in error_str:
            return ModelError(ModelErrorType.INVALID_REQUEST, str(error), retryable=False)

        return ModelError(ModelErrorType.UNKNOWN, str(error), retryable=True)

    def _gpt4all_generate(
        self, prompt: str, max_tokens: Optional[int] = None, temperature: Optional[float] = None, stop: Optional[list[str]] = None, stream: bool = False
    ) -> str:
        mt = max_tokens or config.MAX_TOKENS
        temp = temperature if temperature is not None else config.TEMP

        if stream:
            def gen():
                """Gen."""
                for token in self._model.generate(
                    prompt=prompt,
                    max_tokens=mt,
                    temp=temp,
                    streaming=True,
                ):
                    yield token
                self._retry_stats["success"] += 1
            return gen()
        else:
            result = self._model.generate(
                prompt=prompt,
                max_tokens=mt,
                temp=temp,
                streaming=False,
            )
            self._retry_stats["success"] += 1
            return result

    def _ollama_generate(self, prompt: str, max_tokens, temperature, stop, stream):
        mt = max_tokens or config.MAX_TOKENS
        temp = temperature if temperature is not None else config.TEMP

        payload = {
            "model": self._ollama_model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "num_predict": mt,
                "temperature": temp,
            },
        }

        if stop:
            payload["options"]["stop"] = stop

        if stream:
            def gen():
                """Gen."""
                try:
                    r = requests.post(
                        f"{self._ollama_base}/api/generate",
                        json=payload,
                        stream=True,
                        timeout=self.REQUEST_TIMEOUT,
                    )
                    r.raise_for_status()
                    r.encoding = "utf-8"
                    for line in r.iter_lines(decode_unicode=True):
                        if line:
                            try:
                                data = json.loads(line)
                                text = data.get("response", "")
                                if text:
                                    yield text
                                if data.get("done"):
                                    break
                            except json.JSONDecodeError:
                                continue
                    self._retry_stats["success"] += 1
                except Exception as e:
                    raise e

            return gen()
        else:
            try:
                r = requests.post(
                    f"{self._ollama_base}/api/generate",
                    json=payload,
                    timeout=self.REQUEST_TIMEOUT,
                )
                r.raise_for_status()
                data = r.json()
                self._retry_stats["success"] += 1
                return data.get("response", "")
            except Exception as e:
                raise e

    def _llama_generate(self, prompt: str, max_tokens, temperature, stop, stream):
        if not self._model:
            self._load_llama_cpp("")

        mt = max_tokens or config.MAX_TOKENS
        temp = temperature if temperature is not None else config.TEMP
        stops = stop or []

        if stream:
            def gen():
                """Gen."""
                for chunk in self._model.create_completion(
                    prompt=prompt, max_tokens=mt, temperature=temp,
                    stop=stops, stream=True, echo=False,
                ):
                    text = chunk.get("choices", [{}])[0].get("text", "")
                    if text:
                        yield text
                self._retry_stats["success"] += 1
            return gen()
        else:
            result = self._model.create_completion(
                prompt=prompt, max_tokens=mt, temperature=temp,
                stop=stops, stream=False, echo=False,
            )
            self._retry_stats["success"] += 1
            return result.get("choices", [{}])[0].get("text", "")

    def tokenize(self, text: str) -> list[int]:
        """Tokenize."""
        if self._use_ollama:
            return self._ollama_tokenize(text)
        if not self._model:
            self._load_llama_cpp("")
        return self._model.tokenize(text.encode("utf-8"))

    def _ollama_tokenize(self, text: str) -> list[int]:
        cache_key = hashlib.md5(text.encode()).hexdigest()
        if cache_key in self._token_cache:
            cached_len = self._token_cache[cache_key]
            return list(range(cached_len))

        try:
            r = requests.post(
                f"{self._ollama_base}/api/tokenize",
                json={"model": self._ollama_model, "text": text},
                timeout=10,
            )
            if r.status_code == 200:
                data = r.json()
                tokens = data.get("tokens", [])
                self._token_cache[cache_key] = len(tokens)
                return tokens
        except Exception:
            pass

        estimated = max(1, len(text) // 4)
        self._token_cache[cache_key] = estimated
        return list(range(estimated))

    def count_tokens(self, text: str) -> int:
        """Return the estimated token count for the given text."""
        if self._use_ollama:
            cache_key = hashlib.md5(text.encode()).hexdigest()
            if cache_key in self._token_cache:
                return self._token_cache[cache_key]

            try:
                r = requests.post(
                    f"{self._ollama_base}/api/tokenize",
                    json={"model": self._ollama_model, "text": text},
                    timeout=10,
                )
                if r.status_code == 200:
                    data = r.json()
                    count = len(data.get("tokens", []))
                    self._token_cache[cache_key] = count
                    return count
            except Exception:
                pass

            estimated = max(1, len(text) // 4)
            self._token_cache[cache_key] = estimated
            return estimated

        return len(self.tokenize(text))

    async def agenerate(self, prompt: str, max_tokens=None, temperature=None,
                        stop=None, stream=False, retries=0):
        """Agenerate."""
        import asyncio
        if stream:
            def gen():
                """Gen."""
                return self.generate(prompt, max_tokens, temperature, stop, stream=True, retries=retries)
            return await asyncio.to_thread(gen)
        return await asyncio.to_thread(
            self.generate, prompt, max_tokens, temperature, stop, False, retries
        )

    def get_retry_stats(self) -> dict:
        """Return retry statistics for the current session."""
        return dict(self._retry_stats)

    def unload(self):
        """Release the model from memory."""
        self._model = None
        self._token_cache.clear()

    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass
