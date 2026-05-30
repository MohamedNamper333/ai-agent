import json
import os
import math
from pathlib import Path
from typing import Optional

import config


class VectorStore:
    def __init__(self, db_path: str = ""):
        self.db_path = db_path or str(Path(config.BASE_DIR) / "vector_store.json")
        self.entries: list[dict] = []
        self._loaded = False

    def add(self, text: str, embedding: list[float], metadata: Optional[dict] = None) -> int:
        entry = {
            "id": len(self.entries),
            "text": text,
            "embedding": embedding,
            "metadata": metadata or {},
        }
        self.entries.append(entry)
        return entry["id"]

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        scored = []
        for entry in self.entries:
            score = self._cosine_similarity(query_embedding, entry["embedding"])
            scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"text": e["text"], "score": s, "metadata": e["metadata"]}
            for s, e in scored[:top_k]
        ]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(ai * bi for ai, bi in zip(a, b))
        na = math.sqrt(sum(ai * ai for ai in a))
        nb = math.sqrt(sum(bi * bi for bi in b))
        if na == 0 or nb == 0:
            return 0
        return dot / (na * nb)

    def save(self) -> None:
        try:
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(self.entries, f, ensure_ascii=False)
        except Exception as e:
            print(f"[vector_store] Warning: save failed: {e}")

    def load(self) -> None:
        if self._loaded:
            return
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    self.entries = json.load(f)
            except Exception as e:
                print(f"[vector_store] Warning: load failed: {e}")
        self._loaded = True

    def clear(self) -> None:
        self.entries = []
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
