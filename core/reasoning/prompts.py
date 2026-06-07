"""Few-shot CoT prompts tuned for free models like qwen2.5:7b and
deepseek-flash.  Capped at 5 steps; explicit output format requested;
6-shot examples per level where the model supports them.
"""
from __future__ import annotations

from typing import Optional


class CoTPrompts:
    """Templates for chain-of-thought reasoning."""

    SYSTEM_BASE = (
        "You are a careful, step-by-step reasoning assistant. "
        "Break complex problems into small numbered steps. "
        "For each step, briefly state: (1) your thought, (2) any action you "
        "would take, (3) your confidence in this step (0.0 to 1.0). "
        "After all steps, give a single final answer. "
        "Keep each step under 30 words. Maximum 5 steps."
    )

    OUTPUT_FORMAT = (
        "\n\nOutput format (strict):\n"
        "Step 1: <thought> | <action> | confidence: <0.0-1.0>\n"
        "Step 2: <thought> | <action> | confidence: <0.0-1.0>\n"
        "...\n"
        "Final answer: <concise answer>\n"
    )

    FEW_SHOT_SIMPLE = (
        "Example 1:\n"
        "Q: What is 25% of 80?\n"
        "Step 1: 25% means 0.25 | multiply 0.25 by 80 | confidence: 0.99\n"
        "Step 2: 0.25 * 80 = 20 | state result | confidence: 0.99\n"
        "Final answer: 20\n\n"
        "Example 2:\n"
        "Q: Capital of Japan?\n"
        "Step 1: Recall Asian capitals | lookup Japan | confidence: 0.99\n"
        "Step 2: Tokyo is the capital of Japan | state answer | confidence: 0.99\n"
        "Final answer: Tokyo\n\n"
    )

    FEW_SHOT_MODERATE = (
        "Example 1:\n"
        "Q: Compare Python lists and tuples.\n"
        "Step 1: Identify key dimensions: mutability, syntax, performance, use "
        "cases | set up comparison | confidence: 0.9\n"
        "Step 2: Mutability - lists mutable, tuples immutable | note difference "
        "| confidence: 0.95\n"
        "Step 3: Syntax - lists use [], tuples use () | note syntax | "
        "confidence: 0.95\n"
        "Step 4: Performance - tuples slightly faster (no resize) | note perf "
        "| confidence: 0.85\n"
        "Step 5: Use cases - lists for collections, tuples for records/keys | "
        "conclude | confidence: 0.9\n"
        "Final answer: Lists are mutable, use [], best for changing "
        "collections. Tuples are immutable, use (), faster, best for fixed "
        "records and dict keys.\n\n"
        "Example 2:\n"
        "Q: Summarise the benefits of unit testing in three bullet points.\n"
        "Step 1: Identify the three most important benefits | pick top three | "
        "confidence: 0.9\n"
        "Step 2: Catches regressions early in development | note benefit | "
        "confidence: 0.95\n"
        "Step 3: Documents expected behaviour | note benefit | "
        "confidence: 0.9\n"
        "Step 4: Enables safe refactoring | note benefit | confidence: 0.95\n"
        "Final answer: - Catches regressions early; - Documents expected "
        "behaviour; - Enables safe refactoring.\n\n"
    )

    FEW_SHOT_DEEP = (
        "Example 1:\n"
        "Q: Design a rate limiter for a public API handling 10k req/s.\n"
        "Step 1: Constraints - high QPS, fairness, low latency, simple ops | "
        "identify requirements | confidence: 0.9\n"
        "Step 2: Algorithm choice - token bucket allows bursts, sliding "
        "window smoother | pick token bucket | confidence: 0.85\n"
        "Step 3: Storage - Redis INCR with EXPIRE for atomicity per key | "
        "pick Redis | confidence: 0.9\n"
        "Step 4: Key strategy - per-user or per-IP token buckets | decide "
        "per-user for fairness | confidence: 0.8\n"
        "Step 5: Failure mode - Redis down: fail-open with warning logs | "
        "decide on failure policy | confidence: 0.75\n"
        "Final answer: Token bucket in Redis, per-user keys, 100 req/min "
        "default, fail-open with logging if Redis is unavailable.\n\n"
        "Example 2:\n"
        "Q: Plan a refactor of a 50k-line monolithic Python app into "
        "microservices.\n"
        "Step 1: Map current modules and dependencies | inventory code | "
        "confidence: 0.9\n"
        "Step 2: Identify bounded contexts (auth, billing, reporting) | "
        "domain analysis | confidence: 0.85\n"
        "Step 3: Choose strangler-fig migration over big-bang | decide "
        "strategy | confidence: 0.9\n"
        "Step 4: Extract auth service first, route traffic via API gateway | "
        "phase 1 plan | confidence: 0.85\n"
        "Step 5: Monitor SLOs and rollback triggers per service | decide "
        "observability | confidence: 0.8\n"
        "Final answer: Strangler-fig with auth first, then billing, then "
        "reporting; API gateway fronting; per-service SLOs and rollback "
        "automation.\n\n"
    )

    @classmethod
    def build_prompt(cls, query: str, level: str = "moderate") -> str:
        """Build full CoT prompt for a query at a given reasoning level."""
        if level == "deep":
            few_shot = cls.FEW_SHOT_DEEP
        elif level == "simple":
            few_shot = cls.FEW_SHOT_SIMPLE
        else:
            few_shot = cls.FEW_SHOT_MODERATE
        return few_shot + f"Q: {query}\n" + cls.OUTPUT_FORMAT

    @classmethod
    def build_system_message(cls) -> str:
        return cls.SYSTEM_BASE
