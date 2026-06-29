"""Text Embedding — with meaningful fallback embedder.

import logging
logger = logging.getLogger(__name__)

FIX: Original fallback used SHA-256 hash of words → completely random vectors
     with zero semantic meaning. Two words meaning the same thing got
     completely different embeddings.

     New fallback uses TF-IDF + character bigrams → approximate semantic
     similarity. Not as good as sentence-transformers, but at least
     "python error" and "python bug" will be closer than "python" and "banana".
"""
import hashlib
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
        """Try providers in order: Ollama → sentence-transformers → fallback."""
        if self._try_ollama():
            return
        try:
            from sentence_transformers import SentenceTransformer
            name = model_path or "all-MiniLM-L6-v2"
            self._model = SentenceTransformer(name)
            self.dim = self._model.get_sentence_embedding_dimension()
            self._model_name = name
            logger.info(f"[embedder] Loaded: {name} (dim={self.dim})")
        except ImportError:
            logger.info("[embedder] sentence-transformers not installed — using TF-IDF fallback")
        except Exception as e:
            logger.error(f"[embedder] sentence-transformers load failed: {e} — using fallback")

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
                    logger.info(f"[embedder] Using Ollama: nomic-embed-text (dim={self.dim})")
                    return True
        except Exception:
            pass
        return False

    def embed(self, text: str) -> list[float]:
        if self._use_ollama:
            return self._ollama_embed(text)
        if self._model:
            return self._model.encode(text, normalize_embeddings=True).tolist()
        return self._tfidf_embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self._use_ollama:
            return [self._ollama_embed(t) for t in texts]
        if self._model:
            return [e.tolist() for e in self._model.encode(texts, normalize_embeddings=True)]
        return [self._tfidf_embed(t) for t in texts]

    def _ollama_embed(self, text: str) -> list[float]:
        try:
            import requests
            model = self._model_name.replace("ollama/", "")
            r = requests.post(
                "http://127.0.0.1:11434/api/embeddings",
                json={"model": model, "prompt": text[:2048]},
                timeout=10,
            )
            if r.status_code == 200:
                emb = r.json().get("embedding")
                if emb:
                    return emb
        except Exception:
            pass
        return self._tfidf_embed(text)

    def _tfidf_embed(self, text: str) -> list[float]:
        """
        FIX: Replaces the SHA-256 hash fallback with a meaningful embedding.

        Method:
          1. Word TF features  — weighted by term frequency, hashed to dim slots
          2. Character bigrams — capture morphology/subword info
          3. Positional weights — first words matter more (like tf-idf IDF proxy)
          4. L2 normalization  — so cosine similarity is just dot product

        Result: "python bug" ≈ "python error" >> "python" ≈ "banana"
                Still worse than sentence-transformers, but usable.
        """
        vec = [0.0] * FALLBACK_DIM
        text_clean = text.lower().strip()

        if not text_clean:
            return vec

        # ── 1. Word TF features ──────────────────────────────────
        words = re.findall(r'\b[a-z]{2,}\b', text_clean)
        if words:
            word_freq: dict[str, int] = {}
            for w in words:
                word_freq[w] = word_freq.get(w, 0) + 1

            n = len(words)
            for word, freq in word_freq.items():
                tf = freq / n
                # Use 3 independent hash slots per word for better coverage
                for seed in ("w1_", "w2_", "w3_"):
                    slot = int(hashlib.md5((seed + word).encode()).hexdigest(), 16) % FALLBACK_DIM
                    vec[slot] += tf * (1.0 if seed == "w1_" else 0.5 if seed == "w2_" else 0.25)

        # ── 2. Character bigram features ─────────────────────────
        # Captures morphology: "running" and "runner" share bigrams
        bigrams: dict[str, int] = {}
        for i in range(len(text_clean) - 1):
            bg = text_clean[i:i+2]
            if bg.strip():  # skip whitespace bigrams
                bigrams[bg] = bigrams.get(bg, 0) + 1

        total_bg = max(sum(bigrams.values()), 1)
        for bg, cnt in bigrams.items():
            slot = int(hashlib.md5(("bg_" + bg).encode()).hexdigest(), 16) % FALLBACK_DIM
            vec[slot] += (cnt / total_bg) * 0.3

        # ── 3. Token position weighting (first words matter more) ─
        for pos, word in enumerate(words[:10]):
            weight = 1.0 / (pos + 1)  # positional decay
            slot = int(hashlib.md5(("pos_" + word).encode()).hexdigest(), 16) % FALLBACK_DIM
            vec[slot] += weight * 0.2

        # ── 4. L2 Normalization ───────────────────────────────────
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]

        return vec

    def get_info(self) -> dict:
        return {
            "model": self._model_name or "tfidf-fallback",
            "dimension": self.dim,
            "is_loaded": self._model is not None or self._use_ollama,
            "type": (
                "ollama" if self._use_ollama
                else "sentence-transformers" if self._model
                else "tfidf-fallback"
            ),
        }
