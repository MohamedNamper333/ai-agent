"""Text Embedding - with Ollama API and multiple backend support"""

import hashlib
import json
import math
import re
from typing import Optional

FALLBACK_DIM = 384


class Embedder:
    def __init__(self):
        self._model = None
        self.dim = FALLBACK_DIM
        self._model_name = ""
        self._use_ollama = False

    def load(self, model_path: str = "") -> None:
        if self._try_ollama():
            return

        try:
            from sentence_transformers import SentenceTransformer
            model_name = model_path or "all-MiniLM-L6-v2"
            self._model = SentenceTransformer(model_name)
            self.dim = self._model.get_sentence_embedding_dimension()
            self._model_name = model_name
        except ImportError:
            pass
        except Exception:
            pass

    def _try_ollama(self) -> bool:
        try:
            import requests
            r = requests.post(
                "http://127.0.0.1:11434/api/embeddings",
                json={"model": "nomic-embed-text", "prompt": "test"},
                timeout=3,
            )
            if r.status_code == 200:
                data = r.json()
                if "embedding" in data:
                    self._use_ollama = True
                    self._model_name = "ollama/nomic-embed-text"
                    self.dim = len(data["embedding"])
                    return True
        except Exception:
            pass

        try:
            import requests
            r = requests.post(
                "http://127.0.0.1:11434/api/tags",
                timeout=2,
            )
            if r.status_code == 200:
                models = r.json().get("models", [])
                for m in models:
                    name = m.get("name", "")
                    if "embed" in name or "mini" in name:
                        self._use_ollama = True
                        self._model_name = f"ollama/{name}"
                        self.dim = 768 if "nomic" in name else 384
                        return True
        except Exception:
            pass

        return False

    def embed(self, text: str) -> list[float]:
        if self._use_ollama:
            return self._ollama_embed(text)
        if self._model:
            return self._model.encode(text, normalize_embeddings=True).tolist()
        return self._simple_embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self._use_ollama:
            return [self._ollama_embed(t) for t in texts]
        if self._model:
            embeddings = self._model.encode(texts, normalize_embeddings=True)
            return [e.tolist() for e in embeddings]
        return [self._simple_embed(t) for t in texts]

    def _ollama_embed(self, text: str) -> list[float]:
        try:
            import requests
            r = requests.post(
                "http://127.0.0.1:11434/api/embeddings",
                json={"model": self._model_name.replace("ollama/", ""), "prompt": text[:2048]},
                timeout=10,
            )
            if r.status_code == 200:
                return r.json().get("embedding", self._simple_embed(text))
        except Exception:
            pass
        return self._simple_embed(text)

    def _simple_embed(self, text: str) -> list[float]:
        text = text.lower().strip()
        words = re.findall(r'\w+', text)

        word_vectors = []
        for word in words[:50]:
            h = hashlib.sha256(word.encode()).digest()
            vec = [b / 255.0 for b in h]
            word_vectors.append(vec)

        if not word_vectors:
            return [0.0] * FALLBACK_DIM

        avg = [0.0] * min(len(word_vectors[0]), 32)
        for vec in word_vectors:
            for i in range(min(len(vec), len(avg))):
                avg[i] += vec[i]
        for i in range(len(avg)):
            avg[i] /= len(word_vectors)

        result = []
        while len(result) < FALLBACK_DIM:
            result.extend(avg)
        return result[:FALLBACK_DIM]

    def get_info(self) -> dict:
        return {
            "model": self._model_name or "fallback",
            "dimension": self.dim,
            "is_loaded": self._model is not None or self._use_ollama,
        }
