"""Chain-of-Thought engine.

The engine is intentionally small. It:

1. Builds a single user message that asks the LLM to "think step by
   step" up to ``MAX_STEPS`` steps.
2. Parses the response line-by-line into :class:`ReasoningStep` records.
3. Surfaces the first line that starts with ``Final:`` as the answer.
4. Approximates a confidence score in :attr:`ReasoningChain.confidence`
   so callers can decide whether to retry or escalate to a verifier
   (W2).

The engine works with any object that exposes ``generate(LLMRequest)``.
That includes :class:`core.llm.LLMRouter` and the legacy
``core.model.LLM``.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from core.llm.base import LLMRequest
from core.reasoning.prompts import CoTPrompts


MAX_STEPS = 5
CONFIDENCE_THRESHOLD = 0.7
UNCERTAIN_MARKERS = (
    "i'm not sure",
    "i am not sure",
    "i don't know",
    "not enough information",
    "unclear",
    "cannot determine",
    "maybe",
    "perhaps",
    "possibly",
    "i think",
    "might be",
    "not certain",
    "uncertain",
)


@dataclass
class ReasoningStep:
    index: int
    thought: str
    action: str
    observation: str

    def to_line(self) -> str:
        return CoTPrompts.format_step(
            self.index, self.thought, self.action, self.observation
        )


@dataclass
class ReasoningChain:
    question: str = ""
    steps: list[ReasoningStep] = field(default_factory=list)
    answer: str = ""
    raw: str = ""
    confidence: float = 0.0
    latency_ms: int = 0
    duration_steps: int = 0

    @property
    def ok(self) -> bool:
        return bool(self.answer) and self.confidence >= CONFIDENCE_THRESHOLD


_STEP_HEADER = re.compile(r"^\s*step\s+(\d+)\s*[:.\-]\s*(.*)$", re.IGNORECASE)
_FINAL_HEADER = re.compile(r"^\s*final\s*[:.\-]\s*(.*)$", re.IGNORECASE)


class CoTEngine:
    """Synchronous CoT executor.

    Parameters
    ----------
    router : object
        Anything with a ``generate(LLMRequest)`` method that returns
        ``str | Iterator[str]``. Accepts both :class:`LLMRouter` and the
        legacy ``LLM`` (the router exposes the same shape).
    prompts : CoTPrompts
        Prompt templates. Override for tests.
    max_steps : int
        Hard cap on the number of steps the prompt asks the LLM for.
        The engine itself does not truncate the response.
    """

    def __init__(self, router: Any, prompts: Optional[CoTPrompts] = None, max_steps: int = MAX_STEPS) -> None:
        self.router = router
        self.prompts = prompts or CoTPrompts()
        self.max_steps = max(1, int(max_steps))

    def think(
        self,
        question: str,
        context: str = "",
        level: str = "deep",
    ) -> ReasoningChain:
        chain = ReasoningChain(question=question)
        user_prompt = self.prompts.user_template.format(
            prompt=question.strip(),
            context=(context or "(none)").strip(),
            max_steps=self.max_steps,
        )
        request = LLMRequest(
            prompt=f"{self.prompts.system}\n\n{user_prompt}",
            max_tokens=1024,
            temperature=0.2,
            metadata={"level": level},
        )
        start = time.monotonic()
        try:
            response = self.router.generate(request)
        except Exception as exc:  # pragma: no cover - defensive
            chain.answer = ""
            chain.confidence = 0.0
            chain.raw = f"<router error: {exc}>"
            chain.latency_ms = int((time.monotonic() - start) * 1000)
            return chain

        text = _coerce_to_text(response)
        chain.raw = text
        chain.latency_ms = int((time.monotonic() - start) * 1000)

        chain.steps, final_answer = _parse_response(text)
        chain.answer = final_answer
        chain.duration_steps = len(chain.steps)
        chain.confidence = _estimate_confidence(chain)
        return chain

    async def think_async(
        self,
        question: str,
        context: str = "",
        level: str = "deep",
    ) -> ReasoningChain:
        """Async variant; delegates to ``router.agenerate`` if available."""
        chain = ReasoningChain(question=question)
        user_prompt = self.prompts.user_template.format(
            prompt=question.strip(),
            context=(context or "(none)").strip(),
            max_steps=self.max_steps,
        )
        request = LLMRequest(
            prompt=f"{self.prompts.system}\n\n{user_prompt}",
            max_tokens=1024,
            temperature=0.2,
            metadata={"level": level},
        )
        start = time.monotonic()
        agenerate = getattr(self.router, "agenerate", None)
        try:
            if agenerate is None:
                text = _coerce_to_text(self.router.generate(request))
            else:
                response = await agenerate(request)
                text = _coerce_to_text(response)
        except Exception as exc:  # pragma: no cover - defensive
            chain.raw = f"<router error: {exc}>"
            chain.latency_ms = int((time.monotonic() - start) * 1000)
            return chain
        chain.raw = text
        chain.latency_ms = int((time.monotonic() - start) * 1000)
        chain.steps, final_answer = _parse_response(text)
        chain.answer = final_answer
        chain.duration_steps = len(chain.steps)
        chain.confidence = _estimate_confidence(chain)
        return chain


# -- helpers ----------------------------------------------------------------


def _coerce_to_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    if hasattr(response, "text") and isinstance(getattr(response, "text", None), str):
        return response.text
    if hasattr(response, "__iter__") and not isinstance(response, (list, tuple)):
        try:
            return "".join(str(chunk) for chunk in response)
        except TypeError:
            pass
    if isinstance(response, (list, tuple)):
        return "".join(str(c) for c in response)
    return str(response)


def _parse_response(text: str) -> tuple[list[ReasoningStep], str]:
    steps: list[ReasoningStep] = []
    final_answer = ""

    for raw_line in (text or "").splitlines():
        line = raw_line.rstrip()
        if not line:
            continue
        m_final = _FINAL_HEADER.match(line)
        if m_final:
            final_answer = m_final.group(1).strip()
            continue
        m_step = _STEP_HEADER.match(line)
        if m_step:
            try:
                index = int(m_step.group(1))
            except (TypeError, ValueError):
                index = len(steps) + 1
            payload = m_step.group(2)
            thought, action, observation = _split_step_payload(payload)
            steps.append(
                ReasoningStep(
                    index=index,
                    thought=thought,
                    action=action,
                    observation=observation,
                )
            )

    # If the LLM never emitted a ``Final:`` line, fall back to the
    # last observation, which is a common CoT idiom.
    if not final_answer and steps:
        last = steps[-1]
        if last.observation and last.observation != "(none)":
            final_answer = last.observation
        elif last.thought and last.thought != "(none)":
            final_answer = last.thought

    return steps, final_answer


def _split_step_payload(payload: str) -> tuple[str, str, str]:
    """Extract thought/action/observation from a step body.

    Accepts several separator styles: 'Thought=...; Action=...; ...',
    'Thought: ... Action: ...', or a single free-form string.
    """
    payload = payload.strip()
    if not payload:
        return "", "", ""

    # Try key=value first.
    if "=" in payload:
        thought = _value_after_key(payload, "thought")
        action = _value_after_key(payload, "action")
        observation = _value_after_key(payload, "observation")
        return thought, action, observation

    # Try key: value next.
    thought = _value_after_key(payload, "thought", sep=":")
    action = _value_after_key(payload, "action", sep=":")
    observation = _value_after_key(payload, "observation", sep=":")
    if thought or action or observation:
        return thought, action, observation

    # Free-form: best-effort split on ';' or '.' boundaries.
    return payload, "", ""


def _value_after_key(payload: str, key: str, sep: str = "=") -> str:
    """Pick the substring after ``key<sep>`` up to the next ``;`` or end."""
    needle = f"{key}{sep}"
    lower = payload.lower()
    idx = lower.find(needle)
    if idx < 0:
        return ""
    start = idx + len(needle)
    rest = payload[start:]
    nxt = rest.find(";")
    if nxt < 0:
        return rest.strip()
    return rest[:nxt].strip()


def _estimate_confidence(chain: ReasoningChain) -> float:
    """Heuristic 0-1 score used by the agent to decide whether to retry."""
    if not chain.answer:
        return 0.0
    if any(marker in chain.answer.lower() for marker in UNCERTAIN_MARKERS):
        return 0.2
    if any(marker in chain.raw.lower() for marker in UNCERTAIN_MARKERS):
        return 0.4
    base = 0.5
    # More structured steps → higher confidence, capped at 1.0.
    base += min(0.4, 0.08 * chain.duration_steps)
    if chain.duration_steps == 0:
        return 0.3
    return min(1.0, base)


__all__ = [
    "CoTEngine",
    "ReasoningChain",
    "ReasoningStep",
    "MAX_STEPS",
    "CONFIDENCE_THRESHOLD",
]
