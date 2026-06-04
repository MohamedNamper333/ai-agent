"""core.reasoning — Chain-of-Thought and (later) verifier/reflection.

W1 ships the :class:`CoTEngine` plus its prompts. W2 will add the
verifier and the reflection loop.
"""

from core.reasoning.cot import CoTEngine, ReasoningChain, ReasoningStep
from core.reasoning.prompts import CoTPrompts

__all__ = [
    "CoTEngine",
    "ReasoningChain",
    "ReasoningStep",
    "CoTPrompts",
]
