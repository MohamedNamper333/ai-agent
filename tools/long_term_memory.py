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
        """Load data from storage."""
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
        """Store a new conversation summary with topics and timestamp."""
        self.load()
        self.summaries.append({
            "conversation_id": conversation_id,
            "summary": summary,
            "topics": topics or [],
            "timestamp": datetime.now().isoformat(),
        })
        self._save()

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Full-text search across all conversations and return scored results."""
        self.load()
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored = []
        for s in self.summaries:
            score = 0

            if query_lower in s["summary"].lower():
                score += 3

            if any(query_lower in t.lower() for t in s.get("topics", [])):
                score += 5

            summary_words = set(s["summary"].lower().split())
            word_overlap = len(query_words & summary_words)
            score += word_overlap * 0.5

            for topic in s.get("topics", []):
                topic_words = set(topic.lower().split())
                topic_overlap = len(query_words & topic_words)
                score += topic_overlap * 0.3

            try:
                summary_time = datetime.fromisoformat(s["timestamp"])
                days_old = (datetime.now() - summary_time).days
                recency_bonus = max(0, 10 - days_old * 0.1)
                score += recency_bonus
            except Exception:
                pass

            if score > 0:
                scored.append((score, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]

    def get_context(self, query: str) -> str:
        """Return a formatted recall string for the most relevant summaries."""
        results = self.search(query)
        if not results:
            return ""
        parts = ["[Long-term memory recall]"]
        for score, item in results:
            ts = item.get("timestamp", "")[:10]
            parts.append(f"- [{ts}] {item['summary'][:200]}")
        return "\n".join(parts)

    def get_stats(self) -> dict:
        """Return hit rate, miss count, eviction count, and current size."""
        self.load()
        return {
            "total_summaries": len(self.summaries),
            "unique_topics": len(set(
                t for s in self.summaries for t in s.get("topics", [])
            )),
            "oldest": self.summaries[0]["timestamp"] if self.summaries else "",
            "newest": self.summaries[-1]["timestamp"] if self.summaries else "",
        }

    def delete_summary(self, conversation_id: str) -> bool:
        """Delete all summaries for the given conversation ID."""
        self.load()
        original_len = len(self.summaries)
        self.summaries = [s for s in self.summaries if s.get("conversation_id") != conversation_id]
        if len(self.summaries) < original_len:
            self._save()
            return True
        return False

    def _save(self):
        try:
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(self.summaries, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
