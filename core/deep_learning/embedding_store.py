"""FAISS-backed semantic embedding store for persistent memory search."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

DIM = 384  # sentence-transformers all-MiniLM-L6-v2 dimension


class EmbeddingStore:
    """Semantic vector store using FAISS for fast approximate nearest-neighbor search.

    Falls back to cosine similarity (numpy) when FAISS is unavailable.
    """

    STORE_PATH = Path("learning_data/embedding_store")

    def __init__(self, dim: int = DIM):
        """Initialize embedding store with given vector dimension."""
        self.dim = dim
        self._index = None
        self._metadata: list[dict] = []
        self._embedder = None
        self._use_faiss = False
        self._init_index()
        self._init_embedder()

    def _init_index(self) -> None:
        """Initialize FAISS flat L2 index or numpy fallback."""
        try:
            import faiss
            self._index = faiss.IndexFlatIP(self.dim)  # Inner product = cosine on normalized vecs
            self._use_faiss = True
            logger.info("EmbeddingStore: FAISS initialized (dim=%d)", self.dim)
            self._load()
        except ImportError:
            logger.warning("EmbeddingStore: FAISS not available, using numpy fallback")
            self._vectors: list[np.ndarray] = []

    def _init_embedder(self) -> None:
        """Try to load sentence-transformers embedder."""
        try:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("EmbeddingStore: sentence-transformers loaded")
        except ImportError:
            logger.warning("EmbeddingStore: sentence-transformers not available, using hash embedder")

    def _hash_embed(self, text: str) -> np.ndarray:
        """Fallback embedding using deterministic hash (fast, low quality)."""
        h = hashlib.sha256(text.lower().encode()).digest()
        # Repeat to fill dim
        raw = np.frombuffer(h * ((self.dim // 32) + 1), dtype=np.uint8)[:self.dim]
        vec = raw.astype(np.float32) / 255.0 - 0.5
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def embed(self, text: str) -> np.ndarray:
        """Convert text to a normalized embedding vector."""
        if self._embedder is not None:
            try:
                vec = self._embedder.encode(text, normalize_embeddings=True)
                return vec.astype(np.float32)
            except Exception as exc:
                logger.warning("EmbeddingStore.embed error: %s", exc)
        return self._hash_embed(text)

    def add(self, text: str, metadata: Optional[dict] = None) -> int:
        """Add a text entry and return its index."""
        vec = self.embed(text)
        idx = len(self._metadata)
        entry = {
            "idx": idx,
            "text": text[:500],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **(metadata or {}),
        }
        self._metadata.append(entry)

        if self._use_faiss:
            self._index.add(vec.reshape(1, -1))
        else:
            self._vectors.append(vec)

        if idx % 10 == 0:
            self._save()
        return idx

    def search(self, query: str, k: int = 5) -> list[dict]:
        """Search for k most semantically similar entries to query.

        Returns:
            List of dicts with keys: text, score, ts, and any metadata fields.
        """
        if not self._metadata:
            return []

        q_vec = self.embed(query).reshape(1, -1)
        k = min(k, len(self._metadata))

        try:
            if self._use_faiss and self._index.ntotal > 0:
                scores, indices = self._index.search(q_vec, k)
                results = []
                for score, idx in zip(scores[0], indices[0]):
                    if 0 <= idx < len(self._metadata):
                        entry = dict(self._metadata[idx])
                        entry["score"] = float(score)
                        results.append(entry)
                return results
            else:
                # Numpy cosine similarity fallback
                vecs = np.stack(self._vectors)
                sims = (vecs @ q_vec.T).flatten()
                top_k = np.argsort(sims)[::-1][:k]
                return [
                    {**self._metadata[i], "score": float(sims[i])}
                    for i in top_k
                ]
        except Exception as exc:
            logger.error("EmbeddingStore.search error: %s", exc)
            return []

    def _save(self) -> None:
        """Persist metadata and FAISS index to disk."""
        try:
            self.STORE_PATH.mkdir(parents=True, exist_ok=True)
            with open(self.STORE_PATH / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(self._metadata, f, ensure_ascii=False, indent=2)
            if self._use_faiss:
                import faiss
                faiss.write_index(self._index, str(self.STORE_PATH / "index.faiss"))
        except Exception as exc:
            logger.warning("EmbeddingStore._save error: %s", exc)

    def _load(self) -> None:
        """Load metadata and FAISS index from disk."""
        try:
            meta_path = self.STORE_PATH / "metadata.json"
            idx_path = self.STORE_PATH / "index.faiss"
            if meta_path.exists():
                with open(meta_path, encoding="utf-8") as f:
                    self._metadata = json.load(f)
            if self._use_faiss and idx_path.exists():
                import faiss
                self._index = faiss.read_index(str(idx_path))
                logger.info("EmbeddingStore: loaded %d vectors from disk", self._index.ntotal)
        except Exception as exc:
            logger.warning("EmbeddingStore._load error: %s", exc)

    def get_stats(self) -> dict:
        """Return store statistics."""
        return {
            "total_entries": len(self._metadata),
            "faiss_enabled": self._use_faiss,
            "embedder": "sentence-transformers" if self._embedder else "hash-fallback",
            "dimension": self.dim,
            "index_size": self._index.ntotal if self._use_faiss else len(getattr(self, "_vectors", [])),
        }
