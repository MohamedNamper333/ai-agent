"""core/memory/neural_memory.py — Neural Memory System

Two-layer memory architecture:
  CONSCIOUS (this file):
    - Stores decisions, reasons, outcomes
    - Fast retrieval via semantic similarity
    - Automatically consolidates old memories
    - SQLite-backed for persistence

  SUBCONSCIOUS (obsidian_bridge.py):
    - Long-term knowledge graph
    - Markdown files with wikilinks
    - Survives system restarts
    - Human-readable and editable

The conscious mind queries both layers when making decisions.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────
#  Memory node types
# ─────────────────────────────────────────────
@dataclass
class MemoryNode:
    """A single memory unit — could be a decision, observation, or lesson."""
    node_id: str
    node_type: str           # decision | observation | lesson | plan | outcome
    content: str             # The actual memory content
    context: str             # What was happening when this was stored
    reasoning: str           # WHY this decision was made
    factors: list[str]       # What factors influenced this
    outcome: str             # What happened as a result (filled in later)
    timestamp: float
    importance: float        # 0.0 - 1.0
    accessed_count: int = 0
    last_accessed: float = 0.0
    tags: list[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if not self.last_accessed:
            self.last_accessed = self.timestamp

    def decay_factor(self) -> float:
        """Memory fades with time unless reinforced (Ebbinghaus curve)."""
        age_hours = (time.time() - self.timestamp) / 3600
        base_decay = math.exp(-age_hours / 168)   # Half-life ≈ 168h (1 week)
        reinforcement = min(1.0, self.accessed_count * 0.1)
        return min(1.0, base_decay + reinforcement) * self.importance

    def summary(self) -> str:
        return (
            f"[{self.node_type.upper()}] {self.content[:100]}\n"
            f"  Why: {self.reasoning[:80]}\n"
            f"  Importance: {self.importance:.2f} | "
            f"Accessed: {self.accessed_count}x"
        )


@dataclass
class MemoryQuery:
    """Query structure for memory retrieval."""
    query_text: str
    node_types: Optional[list[str]] = None
    min_importance: float = 0.0
    top_k: int = 5
    include_decayed: bool = False


@dataclass
class MemorySearchResult:
    node: MemoryNode
    relevance_score: float


# ─────────────────────────────────────────────
#  Neural Memory (Conscious Layer)
# ─────────────────────────────────────────────
class NeuralMemory:
    """
    The 'conscious mind' — active working memory with:
      - SQLite persistence
      - Semantic search via TF-IDF vectors
      - Automatic memory consolidation
      - Decision chain tracking (why → what → outcome)
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS memories (
        node_id TEXT PRIMARY KEY,
        node_type TEXT,
        content TEXT,
        context TEXT,
        reasoning TEXT,
        factors TEXT,          -- JSON array
        outcome TEXT,
        timestamp REAL,
        importance REAL,
        accessed_count INTEGER DEFAULT 0,
        last_accessed REAL,
        tags TEXT,             -- JSON array
        embedding TEXT         -- JSON array (TF-IDF vector)
    );
    CREATE INDEX IF NOT EXISTS idx_type ON memories(node_type);
    CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance);
    CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp);
    """

    def __init__(self, db_path: str = ""):
        self._db_path = db_path or str(Path("learning_data") / "neural_memory.db")
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()
        self._embedding_cache: dict[str, list[float]] = {}

    def _init_db(self) -> None:
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.executescript(self.SCHEMA)
        self._conn.commit()

    # ─────────────────────────────────────────
    #  Write operations
    # ─────────────────────────────────────────
    def remember(
        self,
        content: str,
        node_type: str = "observation",
        context: str = "",
        reasoning: str = "",
        factors: Optional[list[str]] = None,
        importance: float = 0.5,
        tags: Optional[list[str]] = None,
    ) -> str:
        """Store a new memory. Returns node_id."""
        node_id = self._generate_id(content)
        embedding = self._embed(content + " " + reasoning)

        self._conn.execute(
            """
            INSERT OR REPLACE INTO memories
            (node_id, node_type, content, context, reasoning,
             factors, outcome, timestamp, importance, accessed_count,
             last_accessed, tags, embedding)
            VALUES (?, ?, ?, ?, ?, ?, '', ?, ?, 0, ?, ?, ?)
            """,
            (
                node_id, node_type, content[:2000], context[:1000],
                reasoning[:1000], json.dumps(factors or []),
                time.time(), min(1.0, max(0.0, importance)),
                time.time(), json.dumps(tags or []),
                json.dumps(embedding),
            ),
        )
        self._conn.commit()
        return node_id

    def remember_decision(
        self,
        decision: str,
        reasoning: str,
        factors: list[str],
        context: str = "",
        importance: float = 0.8,
    ) -> str:
        """
        Remember a decision with full context.
        This is the core of 'why did I decide this?'
        """
        return self.remember(
            content=decision,
            node_type="decision",
            context=context,
            reasoning=reasoning,
            factors=factors,
            importance=importance,
            tags=["decision", "tracked"],
        )

    def update_outcome(self, node_id: str, outcome: str) -> None:
        """Update what actually happened after a decision was made."""
        # Auto-adjust importance based on outcome sentiment
        importance_bump = 0.1 if any(
            w in outcome.lower() for w in ["success", "worked", "correct", "better"]
        ) else -0.05 if any(
            w in outcome.lower() for w in ["fail", "wrong", "mistake", "bad"]
        ) else 0.0

        self._conn.execute(
            """
            UPDATE memories
            SET outcome = ?,
                importance = MIN(1.0, MAX(0.0, importance + ?))
            WHERE node_id = ?
            """,
            (outcome[:1000], importance_bump, node_id),
        )
        self._conn.commit()

    def reinforce(self, node_id: str) -> None:
        """Mark a memory as accessed — slows decay."""
        self._conn.execute(
            """
            UPDATE memories
            SET accessed_count = accessed_count + 1,
                last_accessed = ?,
                importance = MIN(1.0, importance + 0.05)
            WHERE node_id = ?
            """,
            (time.time(), node_id),
        )
        self._conn.commit()

    # ─────────────────────────────────────────
    #  Retrieval
    # ─────────────────────────────────────────
    def recall(self, query: MemoryQuery) -> list[MemorySearchResult]:
        """
        Retrieve relevant memories by semantic similarity.
        Also reinforces accessed memories (use = strengthen).
        """
        query_emb = self._embed(query.query_text)

        sql = "SELECT * FROM memories WHERE importance >= ?"
        params: list = [query.min_importance]

        if query.node_types:
            placeholders = ",".join("?" * len(query.node_types))
            sql += f" AND node_type IN ({placeholders})"
            params.extend(query.node_types)

        rows = self._conn.execute(sql, params).fetchall()
        col_names = [d[0] for d in self._conn.execute("SELECT * FROM memories LIMIT 0").description]

        scored: list[MemorySearchResult] = []
        for row in rows:
            node_dict = dict(zip(col_names, row))
            node = self._row_to_node(node_dict)

            # Skip heavily decayed memories unless explicitly requested
            if not query.include_decayed and node.decay_factor() < 0.1:
                continue

            try:
                stored_emb = json.loads(node_dict.get("embedding", "[]"))
                sim = self._cosine(query_emb, stored_emb)
                # Boost by importance and decay factor
                adjusted = sim * node.decay_factor()
                scored.append(MemorySearchResult(node=node, relevance_score=adjusted))
            except Exception:
                continue

        # Sort and take top_k
        scored.sort(key=lambda r: r.relevance_score, reverse=True)
        results = scored[:query.query_k if hasattr(query, 'query_k') else query.top_k]

        # Reinforce accessed memories
        for r in results:
            self.reinforce(r.node.node_id)

        return results

    def recall_decisions_about(self, topic: str, top_k: int = 5) -> list[MemoryNode]:
        """Quick retrieval of past decisions related to a topic."""
        results = self.recall(MemoryQuery(
            query_text=topic,
            node_types=["decision"],
            top_k=top_k,
        ))
        return [r.node for r in results]

    def ask_self(self, question: str) -> str:
        """
        The agent queries its own memory to answer questions about itself.
        'Why did I decide X?' 'What worked in situation Y?'
        """
        relevant = self.recall(MemoryQuery(query_text=question, top_k=5))
        if not relevant:
            return "No relevant memories found."

        parts = ["Based on my memory:"]
        for r in relevant:
            parts.append(
                f"\n• [{r.node.node_type}] {r.node.content[:150]}"
            )
            if r.node.reasoning:
                parts.append(f"  Reason: {r.node.reasoning[:100]}")
            if r.node.outcome:
                parts.append(f"  Outcome: {r.node.outcome[:80]}")
        return "\n".join(parts)

    # ─────────────────────────────────────────
    #  Consolidation (memory management)
    # ─────────────────────────────────────────
    def consolidate(self) -> dict:
        """
        Compress old, low-importance memories.
        Like sleep — the brain consolidates during downtime.
        """
        cutoff = time.time() - (7 * 24 * 3600)   # 1 week ago
        rows = self._conn.execute(
            "SELECT node_id, importance FROM memories WHERE timestamp < ? AND importance < 0.3",
            (cutoff,),
        ).fetchall()

        deleted = len(rows)
        if rows:
            ids = [r[0] for r in rows]
            self._conn.execute(
                f"DELETE FROM memories WHERE node_id IN ({','.join('?' * len(ids))})", ids
            )
            self._conn.commit()

        total = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        return {"deleted_old": deleted, "total_remaining": total}

    # ─────────────────────────────────────────
    #  Stats
    # ─────────────────────────────────────────
    def get_stats(self) -> dict:
        total = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        by_type = dict(
            self._conn.execute(
                "SELECT node_type, COUNT(*) FROM memories GROUP BY node_type"
            ).fetchall()
        )
        avg_importance = self._conn.execute(
            "SELECT AVG(importance) FROM memories"
        ).fetchone()[0] or 0.0
        return {
            "total_memories": total,
            "by_type": by_type,
            "avg_importance": round(avg_importance, 3),
            "db_path": self._db_path,
        }

    # ─────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────
    def _row_to_node(self, d: dict) -> MemoryNode:
        return MemoryNode(
            node_id=d["node_id"],
            node_type=d["node_type"],
            content=d["content"],
            context=d["context"],
            reasoning=d["reasoning"],
            factors=json.loads(d.get("factors", "[]")),
            outcome=d.get("outcome", ""),
            timestamp=d["timestamp"],
            importance=d["importance"],
            accessed_count=d["accessed_count"],
            last_accessed=d["last_accessed"],
            tags=json.loads(d.get("tags", "[]")),
        )

    def _embed(self, text: str) -> list[float]:
        """TF-IDF embedding — same as fixed embedder."""
        if text in self._embedding_cache:
            return self._embedding_cache[text]

        import re
        import hashlib

        DIM = 256
        vec = [0.0] * DIM
        text_clean = text.lower().strip()
        if not text_clean:
            return vec

        words = re.findall(r'\b[a-z]{2,}\b', text_clean)
        if words:
            freq: dict[str, int] = {}
            for w in words:
                freq[w] = freq.get(w, 0) + 1
            n = len(words)
            for word, cnt in freq.items():
                tf = cnt / n
                for seed in ("a_", "b_", "c_"):
                    slot = int(hashlib.md5((seed + word).encode()).hexdigest(), 16) % DIM
                    vec[slot] += tf * (0.5 if seed != "a_" else 1.0)

        # Normalize
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]

        self._embedding_cache[text] = vec
        if len(self._embedding_cache) > 1000:
            # Evict oldest 200 entries
            keys = list(self._embedding_cache.keys())
            for k in keys[:200]:
                del self._embedding_cache[k]
        return vec

    @staticmethod
    def _cosine(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na and nb else 0.0

    @staticmethod
    def _generate_id(content: str) -> str:
        h = hashlib.sha256(
            f"{content[:100]}{time.time()}".encode()
        ).hexdigest()[:16]
        return f"mem_{h}"

    def close(self) -> None:
        if self._conn:
            self._conn.close()


# Singleton
_neural_memory: Optional[NeuralMemory] = None

def get_neural_memory() -> NeuralMemory:
    global _neural_memory
    if _neural_memory is None:
        _neural_memory = NeuralMemory()
    return _neural_memory
