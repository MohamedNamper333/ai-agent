"""core/reasoning/deductive_engine.py — Deductive Thinking Engine

The 4-step cognitive loop:
  1. ANALYZE   — examine the problem deeply, find patterns & gaps
  2. GENERATE  — produce N candidate plans/solutions
  3. EVALUATE  — predict consequences (short/long term) for each plan
  4. DECIDE    — pick the best plan, store decision in neural memory

This is Tree-of-Thought (ToT) combined with consequence prediction.
NOT speculation — ToT is proven to improve LLM reasoning on complex tasks
(Yao et al., 2023). The consequence prediction loop is a structured
version of what good strategists do naturally.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    from core.llm import LLMRequest, ReasoningLevel
    from core.llm.router import LLMRouter
    _has_router = True
except ImportError:
    _has_router = False

try:
    from core.model import LLM
except ImportError:
    LLM = None  # type: ignore


# ─────────────────────────────────────────────
#  Data structures
# ─────────────────────────────────────────────
@dataclass
class Plan:
    """A candidate plan with predicted consequences."""
    plan_id: str
    description: str
    steps: list[str]
    pros_short_term: list[str]
    cons_short_term: list[str]
    pros_long_term: list[str]
    cons_long_term: list[str]
    scalability_score: float      # 0.0 - 1.0
    risk_score: float             # 0.0 - 1.0 (lower is safer)
    feasibility_score: float      # 0.0 - 1.0
    innovation_score: float       # 0.0 - 1.0
    composite_score: float = 0.0  # Computed

    def __post_init__(self):
        # Weighted composite: feasibility matters most, then scalability
        self.composite_score = (
            self.feasibility_score * 0.35 +
            self.scalability_score * 0.25 +
            (1.0 - self.risk_score) * 0.25 +
            self.innovation_score * 0.15
        )

    def to_summary(self) -> str:
        return (
            f"Plan {self.plan_id}: {self.description[:80]}\n"
            f"  Score: {self.composite_score:.2f} | "
            f"Feasibility: {self.feasibility_score:.2f} | "
            f"Risk: {self.risk_score:.2f} | "
            f"Scalability: {self.scalability_score:.2f}\n"
            f"  Steps: {len(self.steps)}"
        )


@dataclass
class DeductiveResult:
    """Full output of the deductive reasoning process."""
    problem: str
    analysis: str
    plans: list[Plan]
    chosen_plan: Plan
    reasoning: str                 # Why this plan was chosen
    alternative_considered: str    # What was rejected and why
    confidence: float
    latency_ms: float
    should_improve: bool = False   # Is there a better idea we haven't tried?
    improvement_hint: str = ""

    def to_report(self) -> str:
        lines = [
            "═" * 60,
            "DEDUCTIVE REASONING REPORT",
            "═" * 60,
            f"\nPROBLEM:\n{self.problem}\n",
            f"ANALYSIS:\n{self.analysis}\n",
            "PLANS EVALUATED:",
        ]
        for p in self.plans:
            lines.append(p.to_summary())
        lines += [
            f"\nCHOSEN: Plan {self.chosen_plan.plan_id}",
            f"REASONING:\n{self.reasoning}",
            f"\nREJECTED ALTERNATIVES:\n{self.alternative_considered}",
        ]
        if self.should_improve:
            lines.append(f"\n⚡ IMPROVEMENT HINT:\n{self.improvement_hint}")
        lines.append(f"\nConfidence: {self.confidence:.0%} | Time: {self.latency_ms:.0f}ms")
        return "\n".join(lines)


# ─────────────────────────────────────────────
#  Engine
# ─────────────────────────────────────────────
class DeductiveEngine:
    """
    Thinks through problems in 4 steps:
      Analyze → Generate Plans → Evaluate Consequences → Decide

    Designed to surface the best possible solution, not just the first one.
    """

    def __init__(self, model=None, n_plans: int = 3):
        self._model = model
        self._n_plans = n_plans
        self._decision_log: list[DeductiveResult] = []

    def think(self, problem: str, context: str = "") -> DeductiveResult:
        """Full deductive reasoning cycle."""
        start = time.time()

        # Step 1: Deep analysis
        analysis = self._analyze(problem, context)

        # Step 2: Generate N candidate plans
        plans_raw = self._generate_plans(problem, analysis, context)

        # Step 3: Evaluate each plan's consequences
        plans = self._evaluate_plans(problem, plans_raw)

        # Step 4: Decide on best plan
        chosen, reasoning, rejected, confidence = self._decide(problem, plans)

        # Step 5: Self-question — is there something better?
        should_improve, hint = self._self_question(problem, chosen, plans)

        result = DeductiveResult(
            problem=problem,
            analysis=analysis,
            plans=plans,
            chosen_plan=chosen,
            reasoning=reasoning,
            alternative_considered=rejected,
            confidence=confidence,
            latency_ms=(time.time() - start) * 1000,
            should_improve=should_improve,
            improvement_hint=hint,
        )

        self._decision_log.append(result)
        return result

    # ─────────────────────────────────────────
    #  Step 1: Analyze
    # ─────────────────────────────────────────
    def _analyze(self, problem: str, context: str) -> str:
        prompt = f"""You are an expert analyst performing deep problem analysis.

PROBLEM: {problem}
{"CONTEXT: " + context if context else ""}

Perform a structured analysis covering:
1. ROOT CAUSE: What is the actual underlying problem (not just the symptom)?
2. CONSTRAINTS: What limitations, risks, or boundaries must be respected?
3. PATTERNS: What patterns from similar problems apply here?
4. UNKNOWNS: What information is missing that could affect the solution?
5. SUCCESS CRITERIA: How will we know the solution worked?

Be precise, structured, and focus on what's non-obvious. Skip generic observations.
"""
        return self._call_llm(prompt, max_tokens=800)

    # ─────────────────────────────────────────
    #  Step 2: Generate Plans
    # ─────────────────────────────────────────
    def _generate_plans(self, problem: str, analysis: str, context: str) -> list[dict]:
        prompt = f"""You are a strategic planner generating {self._n_plans} distinct solution plans.

PROBLEM: {problem}
ANALYSIS: {analysis}

Generate exactly {self._n_plans} meaningfully different plans. Each plan must differ in:
- Approach (not just in details)
- Risk profile
- Timelines

Respond ONLY with valid JSON array. No text before or after.
Each object must have these exact keys:
{{
  "plan_id": "A",
  "description": "One-line description",
  "steps": ["step1", "step2", "step3"],
  "pros_short_term": ["pro1", "pro2"],
  "cons_short_term": ["con1"],
  "pros_long_term": ["pro1"],
  "cons_long_term": ["con1", "con2"],
  "scalability_score": 0.8,
  "risk_score": 0.3,
  "feasibility_score": 0.9,
  "innovation_score": 0.6
}}
"""
        raw = self._call_llm(prompt, max_tokens=1500)

        # Parse JSON robustly
        try:
            raw_clean = raw.strip()
            if "```" in raw_clean:
                raw_clean = raw_clean.split("```")[1]
                if raw_clean.startswith("json"):
                    raw_clean = raw_clean[4:]
            return json.loads(raw_clean.strip())
        except Exception:
            # Fallback: return a minimal plan if parsing fails
            return [
                {
                    "plan_id": "A",
                    "description": "Direct solution approach",
                    "steps": ["Analyze", "Implement", "Test"],
                    "pros_short_term": ["Fast"],
                    "cons_short_term": ["May not scale"],
                    "pros_long_term": ["Simple to maintain"],
                    "cons_long_term": ["May need refactoring"],
                    "scalability_score": 0.6,
                    "risk_score": 0.4,
                    "feasibility_score": 0.8,
                    "innovation_score": 0.5,
                }
            ]

    # ─────────────────────────────────────────
    #  Step 3: Evaluate consequences
    # ─────────────────────────────────────────
    def _evaluate_plans(self, problem: str, plans_raw: list[dict]) -> list[Plan]:
        plans = []
        for raw in plans_raw:
            try:
                plan = Plan(
                    plan_id=str(raw.get("plan_id", "?")),
                    description=str(raw.get("description", "")),
                    steps=list(raw.get("steps", [])),
                    pros_short_term=list(raw.get("pros_short_term", [])),
                    cons_short_term=list(raw.get("cons_short_term", [])),
                    pros_long_term=list(raw.get("pros_long_term", [])),
                    cons_long_term=list(raw.get("cons_long_term", [])),
                    scalability_score=float(raw.get("scalability_score", 0.5)),
                    risk_score=float(raw.get("risk_score", 0.5)),
                    feasibility_score=float(raw.get("feasibility_score", 0.5)),
                    innovation_score=float(raw.get("innovation_score", 0.5)),
                )
                plans.append(plan)
            except Exception:
                continue
        return plans or [Plan(
            plan_id="fallback", description="Default plan",
            steps=[], pros_short_term=[], cons_short_term=[],
            pros_long_term=[], cons_long_term=[],
            scalability_score=0.5, risk_score=0.5,
            feasibility_score=0.5, innovation_score=0.5,
        )]

    # ─────────────────────────────────────────
    #  Step 4: Decide
    # ─────────────────────────────────────────
    def _decide(
        self, problem: str, plans: list[Plan]
    ) -> tuple[Plan, str, str, float]:
        # Sort by composite score
        ranked = sorted(plans, key=lambda p: p.composite_score, reverse=True)
        best = ranked[0]

        plan_summaries = "\n".join(p.to_summary() for p in ranked)
        prompt = f"""You are making the final decision between these plans:

PROBLEM: {problem}

RANKED PLANS:
{plan_summaries}

Explain in 3-4 sentences:
1. Why Plan {best.plan_id} is the best choice
2. What the key trade-off is
3. What would make a different plan better

Then on a new line: CONFIDENCE: X% (your confidence this is the right choice)
Then: REJECTED: Brief note on why other plans were not chosen
"""
        response = self._call_llm(prompt, max_tokens=400)

        # Extract confidence
        confidence = 0.75
        for line in response.split("\n"):
            if "CONFIDENCE:" in line:
                try:
                    confidence = float(line.split(":")[1].strip().rstrip("%")) / 100
                except Exception:
                    pass

        # Extract rejected note
        rejected = ""
        for line in response.split("\n"):
            if "REJECTED:" in line:
                rejected = line.split(":", 1)[1].strip()

        reasoning = response.split("CONFIDENCE:")[0].strip()
        return best, reasoning, rejected, confidence

    # ─────────────────────────────────────────
    #  Step 5: Self-question
    # ─────────────────────────────────────────
    def _self_question(
        self, problem: str, chosen: Plan, all_plans: list[Plan]
    ) -> tuple[bool, str]:
        """
        Ask: Is there a better idea we haven't considered?
        This is the key differentiator — the agent questions its own choice.
        """
        prompt = f"""You chose Plan {chosen.plan_id}: {chosen.description}

For the problem: {problem}

Ask yourself honestly:
- Is there a fundamentally different approach we haven't considered?
- Are there assumptions in all plans that might be wrong?
- What would a world-class expert do differently?

If yes: describe the better idea in 2 sentences starting with "BETTER:"
If no: respond exactly "OPTIMAL"
"""
        response = self._call_llm(prompt, max_tokens=200)

        if "OPTIMAL" in response.upper() and "BETTER" not in response.upper():
            return False, ""
        if "BETTER:" in response:
            hint = response.split("BETTER:")[1].strip()
            return True, hint
        return False, ""

    # ─────────────────────────────────────────
    #  LLM bridge
    # ─────────────────────────────────────────
    def _call_llm(self, prompt: str, max_tokens: int = 600) -> str:
        try:
            if _has_router:
                router = LLMRouter()
                req = LLMRequest(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    level=ReasoningLevel.DEEP,
                )
                return router.generate(req).text
        except Exception:
            pass
        try:
            if self._model:
                return self._model.generate(prompt, max_tokens=max_tokens) or ""
        except Exception:
            pass
        return "[LLM unavailable]"

    def get_decision_log(self) -> list[dict]:
        return [
            {
                "problem": r.problem[:100],
                "chosen_plan": r.chosen_plan.plan_id,
                "confidence": r.confidence,
                "time_ms": r.latency_ms,
                "had_improvement_hint": r.should_improve,
            }
            for r in self._decision_log
        ]
