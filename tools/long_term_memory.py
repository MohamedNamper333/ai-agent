"""Memory Summarization - long-term memory through conversation summarization"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import config


class LongTermMemory:
    def __init__(self, db_path: str = ""):
        self.db_path = db_path or str(Path(config.BASE_DIR) / "long_term_memory.json")
        self.summaries: list[dict] = []
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    self.summaries = json.load(f)
            except Exception:
                self.summaries = []
        self._loaded = True

    def add_summary(self, conversation_id: str, summary: str, topics: list[str] = None):
        self.load()
        self.summaries.append({
            "conversation_id": conversation_id,
            "summary": summary,
            "topics": topics or [],
            "timestamp": datetime.now().isoformat(),
        })
        self._save()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        self.load()
        query_lower = query.lower()
        scored = []
        for s in self.summaries:
            score = 0
            if query_lower in s["summary"].lower():
                score += 2
            if any(query_lower in t.lower() for t in s.get("topics", [])):
                score += 3
            if query_lower in s["conversation_id"].lower():
                score += 1
            if score > 0:
                scored.append((score, s))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def get_context(self, query: str) -> str:
        results = self.search(query)
        if not results:
            return ""
        parts = ["[Long-term memory recall]"]
        for score, item in results:
            ts = item.get("timestamp", "")[:10]
            parts.append(f"- [{ts}] {item['summary'][:200]}")
        return "\n".join(parts)

    def _save(self):
        try:
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(self.summaries, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
