"""Reasoning Engine — Chain-of-Thought + Deductive Engine.

Pillar 1: DeductiveEngine — Tree-of-Thought with self-questioning
CoT: Chain-of-thought base reasoning
"""
from .cot import CoTEngine, ReasoningChain, ReasoningStep
from .prompts import CoTPrompts

try:
    from .deductive_engine import DeductiveEngine, DeductiveResult, Plan
    __all__ = [
        "CoTEngine", "ReasoningChain", "ReasoningStep", "CoTPrompts",
        "DeductiveEngine", "DeductiveResult", "Plan",
    ]
except ImportError:
    __all__ = ["CoTEngine", "ReasoningChain", "ReasoningStep", "CoTPrompts"]
