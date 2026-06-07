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

    def add_batch(self, texts: list[str], embeddings: list[list[float]], metadatas: Optional[list[dict]] = None) -> list[int]:
        ids = []
        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            meta = metadatas[i] if metadatas else None
            entry = {
                "id": len(self.entries),
                "text": text,
                "embedding": emb,
                "metadata": meta or {},
            }
            self.entries.append(entry)
            ids.append(entry["id"])
        return ids

    def search(self, query_embedding: list[float], top_k: int = 5, min_score: float = 0.0) -> list[dict]:
        if not self.entries:
            return []
        
        try:
            import numpy as np
            embeddings = np.array([e["embedding"] for e in self.entries], dtype=np.float32)
            query = np.array(query_embedding, dtype=np.float32)
            norms = np.linalg.norm(embeddings, axis=1)
            query_norm = np.linalg.norm(query)
            if query_norm == 0:
                return []
            scores = (embeddings @ query) / (norms * query_norm + 1e-10)
            valid_mask = scores >= min_score
            valid_indices = np.where(valid_mask)[0]
            if len(valid_indices) == 0:
                return []
            valid_scores = scores[valid_indices]
            top_idx = valid_indices[np.argsort(-valid_scores)[:top_k]]
            return [
                {"text": self.entries[i]["text"], "score": float(scores[i]),
                 "metadata": self.entries[i]["metadata"], "id": self.entries[i]["id"]}
                for i in top_idx
            ]
        except ImportError:
            scored = []
            for entry in self.entries:
                score = self._cosine_similarity(query_embedding, entry["embedding"])
                if score >= min_score:
                    scored.append((score, entry))
            scored.sort(key=lambda x: x[0], reverse=True)
            return [
                {"text": e["text"], "score": s, "metadata": e["metadata"], "id": e["id"]}
                for s, e in scored[:top_k]
            ]

    def search_with_filter(self, query_embedding: list[float], top_k: int = 5,
                          metadata_filter: Optional[dict] = None) -> list[dict]:
        filtered = self.entries
        if metadata_filter:
            filtered = []
            for entry in self.entries:
                match = True
                for key, value in metadata_filter.items():
                    if entry["metadata"].get(key) != value:
                        match = False
                        break
                if match:
                    filtered.append(entry)

        scored = []
        for entry in filtered:
            score = self._cosine_similarity(query_embedding, entry["embedding"])
            scored.append((score, entry))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {"text": e["text"], "score": s, "metadata": e["metadata"], "id": e["id"]}
            for s, e in scored[:top_k]
        ]

    def delete(self, entry_id: int) -> bool:
        for i, entry in enumerate(self.entries):
            if entry["id"] == entry_id:
                self.entries.pop(i)
                return True
        return False

    def delete_by_source(self, source: str) -> int:
        original_len = len(self.entries)
        self.entries = [e for e in self.entries if e["metadata"].get("source") != source]
        return original_len - len(self.entries)

    def get_entry(self, entry_id: int) -> Optional[dict]:
        for entry in self.entries:
            if entry["id"] == entry_id:
                return entry
        return None

    def get_all_entries(self) -> list[dict]:
        return list(self.entries)

    def get_stats(self) -> dict:
        sources = set()
        for entry in self.entries:
            source = entry["metadata"].get("source", "unknown")
            sources.add(source)

        return {
            "total_entries": len(self.entries),
            "unique_sources": len(sources),
            "sources": list(sources),
        }

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
