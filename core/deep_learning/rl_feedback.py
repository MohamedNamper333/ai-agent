"""Reinforcement Learning feedback engine — learns from user thumbs up/down."""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class RLFeedbackEngine:
    """Lightweight RL engine using reward signals to improve tool selection.

    Uses a simple multi-armed bandit approach (UCB1 algorithm) to learn
    which tools perform best for each task type based on user feedback.
    """

    FEEDBACK_PATH = Path("learning_data/rl_feedback.json")

    def __init__(self):
        """Initialize the RL feedback engine."""
        # Q-values: task_type → tool_name → average reward
        self._q: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        # Visit counts
        self._n: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # Total interactions per task
        self._task_n: dict[str, int] = defaultdict(int)
        self._total_feedback = 0
        self._load()

    def record_feedback(
        self,
        task_type: str,
        tool_name: str,
        feedback: str,  # "positive" | "negative" | "neutral"
        conv_id: Optional[str] = None,
    ) -> None:
        """Record user feedback for a tool used in a specific task type.

        Args:
            task_type: The classified task type (from TaskClassifier).
            tool_name: The tool that was used.
            feedback: User rating — positive (+1), negative (-1), neutral (0).
        """
        reward_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
        reward = reward_map.get(feedback, 0.0)

        self._n[task_type][tool_name] += 1
        self._task_n[task_type] += 1
        self._total_feedback += 1

        # Incremental mean update
        n = self._n[task_type][tool_name]
        old_q = self._q[task_type][tool_name]
        self._q[task_type][tool_name] = old_q + (reward - old_q) / n

        logger.info(
            "RLFeedback: task=%s tool=%s feedback=%s reward=%.1f Q=%.3f",
            task_type, tool_name, feedback, reward, self._q[task_type][tool_name],
        )

        if self._total_feedback % 5 == 0:
            self._save()

    def get_best_tool(self, task_type: str, candidates: list[str]) -> Optional[str]:
        """Return the best tool for a task using UCB1 exploration strategy.

        Args:
            task_type: The task classification.
            candidates: List of available tool names.

        Returns:
            Name of the recommended tool, or None if no data yet.
        """
        if not candidates:
            return None
        if task_type not in self._q or not self._q[task_type]:
            return None  # No data — let the agent decide normally

        import math
        total_n = max(self._task_n[task_type], 1)
        best_tool, best_score = None, float("-inf")

        for tool in candidates:
            q = self._q[task_type].get(tool, 0.0)
            n = max(self._n[task_type].get(tool, 0), 1)
            # UCB1: balance exploration vs exploitation
            ucb = q + math.sqrt(2 * math.log(total_n) / n)
            if ucb > best_score:
                best_score, best_tool = ucb, tool

        return best_tool

    def get_tool_rankings(self, task_type: str) -> list[dict]:
        """Return sorted tool rankings for a task type."""
        if task_type not in self._q:
            return []
        rankings = [
            {
                "tool": tool,
                "q_value": round(q, 4),
                "visits": self._n[task_type].get(tool, 0),
            }
            for tool, q in self._q[task_type].items()
        ]
        return sorted(rankings, key=lambda x: x["q_value"], reverse=True)

    def _save(self) -> None:
        """Persist feedback data to disk."""
        try:
            self.FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "q": {k: dict(v) for k, v in self._q.items()},
                "n": {k: dict(v) for k, v in self._n.items()},
                "task_n": dict(self._task_n),
                "total": self._total_feedback,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            with open(self.FEEDBACK_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as exc:
            logger.warning("RLFeedbackEngine._save error: %s", exc)

    def _load(self) -> None:
        """Load persisted feedback data."""
        try:
            if self.FEEDBACK_PATH.exists():
                with open(self.FEEDBACK_PATH, encoding="utf-8") as f:
                    data = json.load(f)
                for k, v in data.get("q", {}).items():
                    self._q[k].update(v)
                for k, v in data.get("n", {}).items():
                    self._n[k].update(v)
                self._task_n.update(data.get("task_n", {}))
                self._total_feedback = data.get("total", 0)
                logger.info("RLFeedback: loaded %d feedback records", self._total_feedback)
        except Exception as exc:
            logger.warning("RLFeedbackEngine._load error: %s", exc)

    def get_stats(self) -> dict:
        """Return RL engine statistics."""
        return {
            "total_feedback": self._total_feedback,
            "task_types_learned": len(self._q),
            "tools_ranked": sum(len(v) for v in self._q.values()),
        }
