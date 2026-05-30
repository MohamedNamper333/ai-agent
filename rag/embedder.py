from typing import Optional


class Embedder:
    def __init__(self):
        self._model = None

    def load(self, model_path: str = "") -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                model_path or "all-MiniLM-L6-v2"
            )
        except ImportError:
            print("[embedder] sentence-transformers not installed. Using simple fallback.")

    def embed(self, text: str) -> list[float]:
        if self._model:
            return self._model.encode(text, normalize_embeddings=True).tolist()
        return self._simple_embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self._model:
            import numpy as np
            emb = self._model.encode(texts, normalize_embeddings=True)
            return [e.tolist() for e in emb]
        return [self._simple_embed(t) for t in texts]

    @staticmethod
    def _simple_embed(text: str) -> list[float]:
        import hashlib
        h = hashlib.sha256(text.encode()).digest()
        return [b / 255.0 for b in h[:64]]
