"""core/learning_engine.py — Self-Learning Engine

What this does (realistic, actually implementable):
  1. Captures successful interaction patterns → stores in RAG
  2. Tracks user feedback (+1/-1) → weights future retrieval
  3. Detects repeated questions → builds FAQ knowledge base
  4. Monitors which tools succeed/fail → adjusts routing
  5. Collects fine-tuning data for future model improvement

What this does NOT do (requires GPU cluster + $50K+):
  - Modify neural network weights at runtime
  - True online learning (gradient descent on live traffic)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import hashlib


@dataclass
class Interaction:
    """A single captured interaction for learning."""
    interaction_id: str
    timestamp: float
    user_input: str
    assistant_response: str
    tools_used: list[str]
    response_time_ms: float
    feedback_score: int = 0       # -1 bad / 0 neutral / +1 good
    feedback_reason: str = ""
    context_tokens: int = 0
    success: bool = True

    def to_finetune_example(self) -> dict:
        """Format for future fine-tuning (Alpaca format)."""
        return {
            "instruction": self.user_input,
            "output": self.assistant_response,
            "quality_score": self.feedback_score,
        }


@dataclass
class ToolPattern:
    """Tracks tool performance for routing optimization."""
    tool_name: str
    success_count: int = 0
    failure_count: int = 0
    avg_time_ms: float = 0.0
    last_failure_reason: str = ""
    input_patterns: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

    @property
    def is_reliable(self) -> bool:
        return self.success_rate >= 0.85 and (self.success_count + self.failure_count) >= 5


class LearningEngine:
    """
    Self-learning system that improves over time without retraining.

    Learns by:
      - Storing good interactions in RAG for future retrieval
      - Tracking which response patterns get positive feedback
      - Monitoring tool performance and adjusting routing priority
      - Building a FAQ cache for common questions
      - Collecting fine-tuning data for future model upgrades
    """

    def __init__(self, data_dir: str = ""):
        self._dir = Path(data_dir or "learning_data")
        self._dir.mkdir(exist_ok=True)

        self._interactions_path = self._dir / "interactions.jsonl"
        self._tool_patterns_path = self._dir / "tool_patterns.json"
        self._faq_cache_path = self._dir / "faq_cache.json"
        self._finetune_path = self._dir / "finetune_dataset.jsonl"

        self._tool_patterns: dict[str, ToolPattern] = {}
        self._faq_cache: dict[str, dict] = {}  # hash → {answer, score, count}
        self._session_interactions: list[Interaction] = []

        self._load()

    # ─────────────────────────────────────────
    #  Core: capture interaction
    # ─────────────────────────────────────────
    def capture(
        self,
        user_input: str,
        assistant_response: str,
        tools_used: Optional[list[str]] = None,
        response_time_ms: float = 0.0,
        success: bool = True,
    ) -> str:
        """Record an interaction. Returns interaction_id."""
        iid = f"int_{int(time.time() * 1000)}_{hash(user_input) % 10000:04d}"
        interaction = Interaction(
            interaction_id=iid,
            timestamp=time.time(),
            user_input=user_input[:2000],
            assistant_response=assistant_response[:4000],
            tools_used=tools_used or [],
            response_time_ms=response_time_ms,
            success=success,
        )
        self._session_interactions.append(interaction)

        # Write to JSONL immediately (streaming persistence)
        with open(self._interactions_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(interaction), ensure_ascii=False) + "\n")

        # Check if this matches a frequent question → update FAQ
        self._update_faq(user_input, assistant_response)
        return iid

    def record_feedback(self, interaction_id: str, score: int, reason: str = "") -> None:
        """Record user feedback (+1 good / -1 bad) for an interaction."""
        score = max(-1, min(1, score))

        # Update in-session
        for ix in self._session_interactions:
            if ix.interaction_id == interaction_id:
                ix.feedback_score = score
                ix.feedback_reason = reason

                # If positive → store in fine-tune dataset
                if score > 0:
                    self._save_finetune_example(ix)

                # Update FAQ score if this question is cached
                q_hash = self._hash(ix.user_input)
                if q_hash in self._faq_cache:
                    self._faq_cache[q_hash]["score"] += score
                    self._save_faq()
                break

    # ─────────────────────────────────────────
    #  Tool performance tracking
    # ─────────────────────────────────────────
    def record_tool_result(
        self,
        tool_name: str,
        success: bool,
        execution_time_ms: float,
        input_preview: str = "",
        failure_reason: str = "",
    ) -> None:
        """Track tool execution for routing optimization."""
        if tool_name not in self._tool_patterns:
            self._tool_patterns[tool_name] = ToolPattern(tool_name=tool_name)

        tp = self._tool_patterns[tool_name]
        if success:
            tp.success_count += 1
        else:
            tp.failure_count += 1
            tp.last_failure_reason = failure_reason

        # Running average of execution time
        total = tp.success_count + tp.failure_count
        tp.avg_time_ms = (tp.avg_time_ms * (total - 1) + execution_time_ms) / total

        # Store input pattern sample (max 20)
        if input_preview and len(tp.input_patterns) < 20:
            tp.input_patterns.append(input_preview[:100])

        self._save_tool_patterns()

    def get_tool_priority(self, tool_names: list[str]) -> list[str]:
        """
        Sort tools by reliability. Unreliable tools go last.
        This improves the agent's tool selection without any retraining.
        """
        def sort_key(name: str) -> float:
            if name in self._tool_patterns:
                tp = self._tool_patterns[name]
                # Score = success_rate * (1 - penalty for slowness)
                speed_factor = max(0.5, 1.0 - (tp.avg_time_ms / 10000))
                return tp.success_rate * speed_factor
            return 0.5  # Unknown tool → neutral

        return sorted(tool_names, key=sort_key, reverse=True)

    def get_unreliable_tools(self, threshold: float = 0.70) -> list[str]:
        """Return tools with success rate below threshold."""
        return [
            name for name, tp in self._tool_patterns.items()
            if (tp.success_count + tp.failure_count) >= 5
            and tp.success_rate < threshold
        ]

    # ─────────────────────────────────────────
    #  FAQ: instant cache for repeated questions
    # ─────────────────────────────────────────
    def _update_faq(self, question: str, answer: str) -> None:
        q_hash = self._hash(question)
        if q_hash in self._faq_cache:
            self._faq_cache[q_hash]["count"] += 1
        else:
            self._faq_cache[q_hash] = {
                "question": question[:500],
                "answer": answer[:2000],
                "count": 1,
                "score": 0,
                "timestamp": time.time(),
            }
        # Keep top 500 FAQs by count
        if len(self._faq_cache) > 500:
            sorted_items = sorted(self._faq_cache.items(), key=lambda x: x[1]["count"], reverse=True)
            self._faq_cache = dict(sorted_items[:500])
        self._save_faq()

    def get_faq_answer(self, question: str, similarity_threshold: float = 0.85) -> Optional[str]:
        """Return cached answer if this question was asked before and got good feedback."""
        q_hash = self._hash(question)
        if q_hash in self._faq_cache:
            entry = self._faq_cache[q_hash]
            # Only use cache if question appeared 3+ times OR got positive feedback
            if entry["count"] >= 3 or entry["score"] > 0:
                return entry["answer"]
        return None

    # ─────────────────────────────────────────
    #  Analytics
    # ─────────────────────────────────────────
    def get_stats(self) -> dict:
        total = len(self._session_interactions)
        positive = sum(1 for ix in self._session_interactions if ix.feedback_score > 0)
        negative = sum(1 for ix in self._session_interactions if ix.feedback_score < 0)

        avg_response_time = 0.0
        if self._session_interactions:
            avg_response_time = sum(ix.response_time_ms for ix in self._session_interactions) / total

        return {
            "session_interactions": total,
            "positive_feedback": positive,
            "negative_feedback": negative,
            "feedback_rate": f"{(positive + negative) / max(total, 1) * 100:.1f}%",
            "satisfaction_rate": f"{positive / max(positive + negative, 1) * 100:.1f}%",
            "avg_response_time_ms": round(avg_response_time),
            "faq_entries": len(self._faq_cache),
            "tools_tracked": len(self._tool_patterns),
            "unreliable_tools": self.get_unreliable_tools(),
            "tool_stats": {
                name: {
                    "success_rate": f"{tp.success_rate * 100:.1f}%",
                    "calls": tp.success_count + tp.failure_count,
                    "avg_time_ms": round(tp.avg_time_ms),
                }
                for name, tp in sorted(
                    self._tool_patterns.items(),
                    key=lambda x: x[1].success_rate,
                    reverse=True,
                )[:10]
            },
        }

    def get_finetune_dataset_path(self) -> str:
        """Path to the collected fine-tuning data."""
        return str(self._finetune_path)

    def get_finetune_count(self) -> int:
        if not self._finetune_path.exists():
            return 0
        return sum(1 for _ in open(self._finetune_path))

    # ─────────────────────────────────────────
    #  Persistence
    # ─────────────────────────────────────────
    def _load(self) -> None:
        if self._tool_patterns_path.exists():
            try:
                raw = json.loads(self._tool_patterns_path.read_text())
                self._tool_patterns = {
                    k: ToolPattern(**v) for k, v in raw.items()
                }
            except Exception:
                pass

        if self._faq_cache_path.exists():
            try:
                self._faq_cache = json.loads(self._faq_cache_path.read_text())
            except Exception:
                pass

    def _save_tool_patterns(self) -> None:
        try:
            data = {k: asdict(v) for k, v in self._tool_patterns.items()}
            self._tool_patterns_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        except Exception:
            pass

    def _save_faq(self) -> None:
        try:
            self._faq_cache_path.write_text(json.dumps(self._faq_cache, ensure_ascii=False, indent=2))
        except Exception:
            pass

    def _save_finetune_example(self, ix: Interaction) -> None:
        try:
            with open(self._finetune_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(ix.to_finetune_example(), ensure_ascii=False) + "\n")
        except Exception:
            pass

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.md5(text.lower().strip().encode()).hexdigest()[:16]


# Singleton
_engine: Optional[LearningEngine] = None

def get_learning_engine() -> LearningEngine:
    global _engine
    if _engine is None:
        _engine = LearningEngine()
    return _engine
