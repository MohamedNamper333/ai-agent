"""Prompt templates for the Chain-of-Thought engine.

Centralising the few-shot examples here keeps the engine logic
focused on the algorithm and lets prompts evolve without touching
``cot.py``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CoTPrompts:
    """Few-shot templates used by :class:`CoTEngine`.

    The engine sends ``system`` and ``user`` to the router. ``format_step``
    is a deterministic helper used to render each step from a thought /
    action / observation triple.
    """

    system: str = (
        "You are a careful, step-by-step problem solver. "
        "For every task you must:\n"
        "1. Decompose the question into atomic steps.\n"
        "2. For each step, state a thought, an action if needed, and an observation.\n"
        "3. After the steps, give the final answer prefixed with 'Final:'.\n"
        "If you are uncertain, say 'I'm not sure' and explain the gap."
    )

    user_template: str = (
        "Question:\n{prompt}\n\n"
        "Context:\n{context}\n\n"
        "Reason step by step. Cap at {max_steps} steps. "
        "Each step must be on a single line, beginning with 'Step N:'."
    )

    step_template: str = (
        "Step {index}: Thought={thought}; Action={action}; Observation={observation}"
    )

    final_template: str = "Final: {answer}"

    @staticmethod
    def format_step(index: int, thought: str, action: str = "", observation: str = "") -> str:
        return CoTPrompts.step_template.format(
            index=index,
            thought=thought or "(none)",
            action=action or "(none)",
            observation=observation or "(none)",
        )


__all__ = ["CoTPrompts"]
