"""Chain-of-Thought Engine.

Workflow:
  1. Build a prompt with few-shot examples for the requested level.
  2. Call the LLM router (which picks a free model).
  3. Parse the response into a ReasoningChain.
  4. Return the chain (downstream code can take just .final_answer).
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from core.llm import (
    LLMRequest,
    ReasoningLevel,
    normalize_level,
)


logger = logging.getLogger(__name__)


MAX_STEPS = 5
CONFIDENCE_THRESHOLD = 0.7


@dataclass
class ReasoningStep:
    """A single step in a CoT chain."""

    step_number: int
    thought: str
    action: str
    confidence: float

    def is_high_confidence(self) -> bool:
        """Is high confidence."""
        return self.confidence >= CONFIDENCE_THRESHOLD


@dataclass
class ReasoningChain:
    """Full chain-of-thought output for a query."""

    query: str
    steps: list[ReasoningStep] = field(default_factory=list)
    final_answer: str = ""
    total_confidence: float = 0.0
    level: ReasoningLevel = ReasoningLevel.MODERATE
    model: str = ""
    latency_ms: float = 0.0
    raw: str = ""

    @property
    def step_count(self) -> int:
        """Step count."""
        return len(self.steps)

    @property
    def avg_confidence(self) -> float:
        """Avg confidence."""
        if not self.steps:
            return 0.0
        return sum(s.confidence for s in self.steps) / len(self.steps)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain dict representation of this object."""
        return {
            "query": self.query,
            "steps": [
                {
                    "step_number": s.step_number,
                    "thought": s.thought,
                    "action": s.action,
                    "confidence": s.confidence,
                }
                for s in self.steps
            ],
            "final_answer": self.final_answer,
            "total_confidence": self.total_confidence,
            "avg_confidence": self.avg_confidence,
            "level": self.level.value,
            "model": self.model,
            "latency_ms": self.latency_ms,
        }


_STEP_RE = re.compile(
    r"Step\s+(\d+)\s*:\s*(.+?)\s*\|\s*(.+?)\s*\|\s*confidence\s*:\s*([0-9]*\.?[0-9]+)",
    re.IGNORECASE | re.DOTALL,
)
_FINAL_RE = re.compile(
    r"Final\s+answer\s*:\s*(.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)


class CoTEngine:
    """Engine that prompts an LLM to produce a chain of thought."""

    def __init__(self, llm_router: Any = None, max_steps: int = MAX_STEPS):
        from .prompts import CoTPrompts

        self.prompts = CoTPrompts
        self.llm = llm_router
        self.max_steps = max(1, int(max_steps))

    def _resolve_llm(self) -> Any:
        if self.llm is not None:
            return self.llm
        from core.llm import LLMRouter

        self.llm = LLMRouter()
        return self.llm

    def think(
        self,
        query: str,
        level: Union[ReasoningLevel, str] = ReasoningLevel.MODERATE,
        system: Optional[str] = None,
    ) -> ReasoningChain:
        """Run CoT on a query, return a ReasoningChain.

        Args:
            query: question / task to reason about
            level: simple | moderate | deep (drives few-shot template + model)
            system: optional system message override

        Returns:
            ReasoningChain with steps, final_answer, total_confidence
        """
        lvl = normalize_level(level)
        prompt = self.prompts.build_prompt(query, level=lvl.value)
        system_msg = system or self.prompts.build_system_message()

        llm = self._resolve_llm()
        request = LLMRequest(
            prompt=prompt,
            system=system_msg,
            max_tokens=1500,
            temperature=0.3,
            level=lvl,
        )

        start = time.time()
        try:
            response = llm.generate_full(request)
            text = response.text or ""
        except Exception as e:
            logger.error("CoT LLM call failed: %s", e)
            return ReasoningChain(
                query=query,
                final_answer=f"[CoT failed: {e}]",
                level=lvl,
                latency_ms=(time.time() - start) * 1000,
            )
        latency_ms = (time.time() - start) * 1000

        return self._parse_response(
            query=query,
            text=text,
            level=lvl,
            model=response.model,
            latency_ms=latency_ms,
        )

    def _parse_response(
        self,
        query: str,
        text: str,
        level: ReasoningLevel,
        model: str,
        latency_ms: float,
    ) -> ReasoningChain:
        steps: list[ReasoningStep] = []
        total_confidence = 0.0
        for m in _STEP_RE.finditer(text or ""):
            try:
                step_num = int(m.group(1))
            except ValueError:
                continue
            thought = m.group(2).strip()
            action = m.group(3).strip()
            try:
                conf = float(m.group(4))
            except ValueError:
                conf = 0.5
            conf = max(0.0, min(1.0, conf))
            steps.append(
                ReasoningStep(
                    step_number=step_num,
                    thought=thought,
                    action=action,
                    confidence=conf,
                )
            )
            total_confidence += conf
        steps = steps[: self.max_steps]

        final_answer = ""
        fa = _FINAL_RE.search(text or "")
        if fa:
            final_answer = fa.group(1).strip()
        else:
            lines = [ln.strip() for ln in (text or "").strip().splitlines() if ln.strip()]
            final_answer = lines[-1] if lines else (text or "").strip()

        return ReasoningChain(
            query=query,
            steps=steps,
            final_answer=final_answer,
            total_confidence=total_confidence,
            level=level,
            model=model,
            latency_ms=latency_ms,
            raw=text or "",
        )

    def think_simple(self, query: str) -> str:
        """Think simple."""
        return self.think(query, level=ReasoningLevel.SIMPLE).final_answer

    def think_moderate(self, query: str) -> str:
        """Think moderate."""
        return self.think(query, level=ReasoningLevel.MODERATE).final_answer

    def think_deep(self, query: str) -> str:
        """Think deep."""
        return self.think(query, level=ReasoningLevel.DEEP).final_answer
