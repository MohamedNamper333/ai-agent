import json
import sys
import time
from typing import Optional, Generator
from pathlib import Path

import config

try:
    import requests
except ImportError:
    requests = None


class LLM:
    def __init__(self, backend: str = "auto"):
        self._backend = backend
        self._model = None
        self._model_path = ""
        self._ollama_base = "http://127.0.0.1:11434"
        self._ollama_model = ""
        self._use_ollama = False

    @property
    def is_loaded(self) -> bool:
        return self._use_ollama or self._model is not None

    def load(self, model_path: str = "") -> None:
        path = model_path or config.MODEL_PATH

        if self._backend == "ollama" or (self._backend == "auto" and self._detect_ollama()):
            self._setup_ollama(path)
            return

        self._load_llama_cpp(path)

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
    ):
        if self._use_ollama:
            return self._ollama_generate(prompt, max_tokens, temperature, stop, stream)
        return self._llama_generate(prompt, max_tokens, temperature, stop, stream)

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
                try:
                    r = requests.post(
                        f"{self._ollama_base}/api/generate",
                        json=payload,
                        stream=True,
                        timeout=300,
                    )
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
                except Exception as e:
                    yield f"\n[Ollama error: {e}]"

            return gen()
        else:
            try:
                r = requests.post(
                    f"{self._ollama_base}/api/generate",
                    json=payload,
                    timeout=300,
                )
                r.raise_for_status()
                data = r.json()
                return data.get("response", "")
            except Exception as e:
                return f"[Ollama error: {e}]"

    def _llama_generate(self, prompt: str, max_tokens, temperature, stop, stream):
        if not self._model:
            self._load_llama_cpp("")

        mt = max_tokens or config.MAX_TOKENS
        temp = temperature if temperature is not None else config.TEMP
        stops = stop or []

        if stream:
            def gen():
                for chunk in self._model.create_completion(
                    prompt=prompt, max_tokens=mt, temperature=temp,
                    stop=stops, stream=True, echo=False,
                ):
                    text = chunk.get("choices", [{}])[0].get("text", "")
                    if text:
                        yield text
            return gen()
        else:
            result = self._model.create_completion(
                prompt=prompt, max_tokens=mt, temperature=temp,
                stop=stops, stream=False, echo=False,
            )
            return result.get("choices", [{}])[0].get("text", "")

    def tokenize(self, text: str) -> list[int]:
        if self._use_ollama:
            return [len(text)]
        if not self._model:
            self._load_llama_cpp("")
        return self._model.tokenize(text.encode("utf-8"))

    def count_tokens(self, text: str) -> int:
        return len(self.tokenize(text))

    def unload(self):
        self._model = None

    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass
