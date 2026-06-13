"""tools/agent_swarm.py — Agent Swarm System

A swarm of specialized agents that:
  1. Decompose complex tasks automatically
  2. Run agents in PARALLEL (ThreadPoolExecutor)
  3. Each agent has a specialized role + different "temperature" mindset
  4. Results are synthesized by a Coordinator agent
  5. Agents can spawn sub-swarms for recursive tasks
  6. Integrates with LearningEngine to track which agents perform best

Swarm Patterns:
  - PARALLEL:    All agents work simultaneously → fastest
  - PIPELINE:    Agent A output feeds Agent B → most coherent
  - DEBATE:      Agents argue, then vote → most accurate
  - RECURSIVE:   Agent spawns sub-agents for sub-tasks → most thorough
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

try:
    from core.model import LLM
except ImportError:
    LLM = None  # type: ignore

try:
    from core.learning_engine import get_learning_engine
except ImportError:
    get_learning_engine = None  # type: ignore


# ─────────────────────────────────────────────
#  Agent definitions
# ─────────────────────────────────────────────
@dataclass
class SwarmAgent:
    name: str
    role: str
    system_prompt: str
    temperature: float = 0.7
    max_tokens: int = 1500
    timeout_s: int = 60
    priority: int = 1          # Higher = runs first in PIPELINE mode
    _model: Any = field(default=None, repr=False)

    def __post_init__(self):
        if self._model is None and LLM is not None:
            self._model = LLM()

    def run(self, task: str, context: str = "") -> "AgentResult":
        start = time.time()
        prompt = self._build_prompt(task, context)
        try:
            if self._model is None:
                return AgentResult(
                    agent_name=self.name,
                    role=self.role,
                    output="[No model available]",
                    success=False,
                    time_ms=0,
                )
            response = self._model.generate(prompt, max_tokens=self.max_tokens)
            elapsed = (time.time() - start) * 1000
            return AgentResult(
                agent_name=self.name,
                role=self.role,
                output=response or "",
                success=True,
                time_ms=elapsed,
            )
        except Exception as exc:
            elapsed = (time.time() - start) * 1000
            return AgentResult(
                agent_name=self.name,
                role=self.role,
                output="",
                success=False,
                error=str(exc),
                time_ms=elapsed,
            )

    def _build_prompt(self, task: str, context: str) -> str:
        parts = [
            f"<|system|>\n{self.system_prompt}",
            "\nBe precise, actionable, and expert-level. No padding.",
        ]
        if context:
            parts.append(f"\n## Context from other agents\n{context}")
        parts.append(f"\n<|user|>\n{task}\n<|assistant|>\n")
        return "\n".join(parts)


@dataclass
class AgentResult:
    agent_name: str
    role: str
    output: str
    success: bool
    time_ms: float
    error: str = ""
    confidence: float = 1.0   # Future: LLM self-scores its own confidence

    def summary(self) -> str:
        status = "✓" if self.success else "✗"
        return f"[{status} {self.agent_name} | {self.time_ms:.0f}ms]\n{self.output}"


@dataclass
class SwarmResult:
    task: str
    pattern: str
    agent_results: list[AgentResult]
    synthesis: str
    total_time_ms: float
    success_count: int
    failure_count: int

    def to_text(self) -> str:
        return self.synthesis

    def to_detailed(self) -> str:
        lines = [
            f"## Swarm Result ({self.pattern})",
            f"Task: {self.task[:100]}",
            f"Agents: {self.success_count} succeeded, {self.failure_count} failed",
            f"Time: {self.total_time_ms:.0f}ms",
            "---",
            "## Synthesis",
            self.synthesis,
            "---",
            "## Individual Agent Outputs",
        ]
        for r in self.agent_results:
            lines.append(r.summary())
        return "\n\n".join(lines)


# ─────────────────────────────────────────────
#  Swarm Orchestrator
# ─────────────────────────────────────────────
class AgentSwarm:
    """
    Orchestrates a swarm of specialized agents.

    Usage:
        swarm = AgentSwarm()

        # Fast parallel analysis
        result = swarm.run_parallel("Analyze this code: ...", agents=["analyst", "security", "optimizer"])

        # Debate for accurate answers
        result = swarm.run_debate("Is GraphQL better than REST for this use case?")

        # Full pipeline for complex tasks
        result = swarm.run_pipeline("Design a microservices architecture for an e-commerce platform")
    """

    # Built-in specialist agents
    BUILTIN_AGENTS: dict[str, dict] = {
        "analyst": {
            "name": "The Analyst",
            "role": "Deep analysis and pattern recognition",
            "system_prompt": (
                "You are The Analyst — a specialist in deep analysis, pattern recognition, "
                "and extracting hidden insights from data, code, and systems. "
                "You think in terms of root causes, not symptoms. "
                "You always quantify your findings and flag anomalies. "
                "Format: numbered findings, each with evidence and impact score (1-10)."
            ),
            "temperature": 0.3,
        },
        "architect": {
            "name": "The Architect",
            "role": "System design and architecture",
            "system_prompt": (
                "You are The Architect — a specialist in system design, scalability, "
                "and long-term technical strategy. "
                "You think in terms of trade-offs, constraints, and future extensibility. "
                "You always consider: maintainability, scalability, cost, and security. "
                "Format: architectural diagram in text + decision rationale for each choice."
            ),
            "temperature": 0.4,
        },
        "security": {
            "name": "The Security Expert",
            "role": "Threat modeling and vulnerability detection",
            "system_prompt": (
                "You are The Security Expert — a specialist in threat modeling, "
                "vulnerability detection, and security architecture. "
                "You think like an attacker to defend like a defender. "
                "You always consider: OWASP top 10, data exposure, authentication flows, "
                "and supply chain risks. "
                "Format: threat list with severity (Critical/High/Medium/Low) and mitigations."
            ),
            "temperature": 0.2,
        },
        "optimizer": {
            "name": "The Optimizer",
            "role": "Performance, cost, and efficiency",
            "system_prompt": (
                "You are The Optimizer — a specialist in performance optimization, "
                "cost reduction, and efficiency improvements. "
                "You think in terms of bottlenecks, Big-O complexity, and resource utilization. "
                "You always measure before optimizing and validate improvements. "
                "Format: ranked optimization opportunities with estimated impact."
            ),
            "temperature": 0.3,
        },
        "critic": {
            "name": "The Critic",
            "role": "Devil's advocate and risk assessment",
            "system_prompt": (
                "You are The Critic — a specialist in finding flaws, risks, and blind spots "
                "in plans, code, and systems. "
                "Your job is to DISAGREE and find what others miss. "
                "You are skeptical of assumptions and demand evidence. "
                "Format: challenges and risks, ranked by probability × impact."
            ),
            "temperature": 0.8,
        },
        "researcher": {
            "name": "The Researcher",
            "role": "Knowledge synthesis and best practices",
            "system_prompt": (
                "You are The Researcher — a specialist in knowledge synthesis, "
                "best practices, and connecting ideas across domains. "
                "You draw on patterns from successful systems and academic research. "
                "You always cite principles and patterns, not just opinions. "
                "Format: synthesized recommendations with supporting evidence."
            ),
            "temperature": 0.5,
        },
        "coder": {
            "name": "The Coder",
            "role": "Code implementation and review",
            "system_prompt": (
                "You are The Coder — a specialist in clean, efficient, and maintainable code. "
                "You think in terms of SOLID principles, design patterns, and testability. "
                "You always write production-ready code with error handling. "
                "Format: working code with inline comments for non-obvious parts."
            ),
            "temperature": 0.2,
        },
        "strategist": {
            "name": "The Strategist",
            "role": "Business and product strategy",
            "system_prompt": (
                "You are The Strategist — a specialist in business strategy, "
                "product thinking, and long-term competitive positioning. "
                "You think in terms of market forces, user needs, and sustainable advantage. "
                "Format: strategic options matrix with trade-offs and recommended path."
            ),
            "temperature": 0.6,
        },
    }

    def __init__(self, model=None, max_workers: int = 4):
        self._model = model
        self._max_workers = max_workers
        self._agents: dict[str, SwarmAgent] = {}
        self._load_builtin_agents()
        self._results_history: list[SwarmResult] = []

    def _load_builtin_agents(self) -> None:
        for key, cfg in self.BUILTIN_AGENTS.items():
            self._agents[key] = SwarmAgent(
                name=cfg["name"],
                role=cfg["role"],
                system_prompt=cfg["system_prompt"],
                temperature=cfg.get("temperature", 0.7),
                _model=self._model,
            )

    def add_agent(self, key: str, agent: SwarmAgent) -> None:
        """Register a custom agent."""
        self._agents[key] = agent

    # ─────────────────────────────────────────
    #  Swarm Patterns
    # ─────────────────────────────────────────
    def run_parallel(
        self,
        task: str,
        agent_keys: Optional[list[str]] = None,
        timeout_s: int = 90,
    ) -> SwarmResult:
        """
        Run all agents simultaneously in parallel.
        Fastest pattern. Best for: analysis, brainstorming, review.
        """
        keys = agent_keys or list(self._agents.keys())
        agents = [self._agents[k] for k in keys if k in self._agents]
        if not agents:
            return self._empty_result(task, "PARALLEL")

        start = time.time()
        results: list[AgentResult] = []

        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = {executor.submit(agent.run, task): agent for agent in agents}
            for future in as_completed(futures, timeout=timeout_s):
                try:
                    results.append(future.result(timeout=5))
                except FuturesTimeout:
                    agent = futures[future]
                    results.append(AgentResult(
                        agent_name=agent.name, role=agent.role,
                        output="", success=False, error="timeout", time_ms=timeout_s * 1000,
                    ))
                except Exception as e:
                    agent = futures[future]
                    results.append(AgentResult(
                        agent_name=agent.name, role=agent.role,
                        output="", success=False, error=str(e), time_ms=0,
                    ))

        synthesis = self._synthesize(task, results, "PARALLEL")
        total_ms = (time.time() - start) * 1000
        return self._make_result(task, "PARALLEL", results, synthesis, total_ms)

    def run_pipeline(
        self,
        task: str,
        agent_keys: Optional[list[str]] = None,
    ) -> SwarmResult:
        """
        Sequential pipeline: each agent sees previous agents' outputs.
        Most coherent. Best for: design tasks, code generation, planning.
        """
        keys = agent_keys or ["researcher", "architect", "coder", "security", "critic"]
        agents = [self._agents[k] for k in keys if k in self._agents]
        if not agents:
            return self._empty_result(task, "PIPELINE")

        start = time.time()
        results: list[AgentResult] = []
        accumulated_context = ""

        for agent in agents:
            result = agent.run(task, context=accumulated_context)
            results.append(result)
            if result.success and result.output:
                accumulated_context += f"\n\n### {agent.name} ({agent.role})\n{result.output}"

        synthesis = self._synthesize(task, results, "PIPELINE")
        total_ms = (time.time() - start) * 1000
        return self._make_result(task, "PIPELINE", results, synthesis, total_ms)

    def run_debate(
        self,
        task: str,
        agent_keys: Optional[list[str]] = None,
        rounds: int = 2,
    ) -> SwarmResult:
        """
        Agents argue, then synthesize.
        Most accurate for controversial decisions.
        Best for: architecture decisions, "which approach is better?", risk assessment.
        """
        keys = agent_keys or ["analyst", "critic", "architect", "optimizer"]
        agents = [self._agents[k] for k in keys if k in self._agents]
        if not agents:
            return self._empty_result(task, "DEBATE")

        start = time.time()
        all_results: list[AgentResult] = []

        # Round 1: Initial positions (parallel)
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = [executor.submit(agent.run, task) for agent in agents]
            round1 = [f.result(timeout=60) for f in futures if not f.exception()]
        all_results.extend(round1)

        # Round 2: Rebuttals (each agent sees others' positions)
        if rounds >= 2:
            round1_context = "\n\n".join(
                f"### {r.agent_name}\n{r.output}" for r in round1 if r.success
            )
            rebuttal_task = f"Other experts have weighed in. Refine your position:\n\n{round1_context}\n\nOriginal question: {task}"
            with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
                futures = [executor.submit(agent.run, rebuttal_task) for agent in agents]
                round2 = [f.result(timeout=60) for f in futures if not f.exception()]
            all_results.extend(round2)

        synthesis = self._synthesize(task, all_results, "DEBATE")
        total_ms = (time.time() - start) * 1000
        return self._make_result(task, "DEBATE", all_results, synthesis, total_ms)

    def run_auto(self, task: str) -> SwarmResult:
        """
        Automatically choose the best swarm pattern based on task characteristics.
        """
        task_lower = task.lower()

        # Code generation → pipeline
        if any(w in task_lower for w in ["write code", "implement", "build", "create function", "fix bug"]):
            return self.run_pipeline(task, ["researcher", "coder", "security", "critic"])

        # Architecture / design decision → debate
        if any(w in task_lower for w in ["should i", "which is better", "compare", "choose between", "architect"]):
            return self.run_debate(task)

        # Analysis / review → parallel (all agents at once)
        return self.run_parallel(task)

    # ─────────────────────────────────────────
    #  Synthesis
    # ─────────────────────────────────────────
    def _synthesize(self, task: str, results: list[AgentResult], pattern: str) -> str:
        """Coordinator synthesizes all agent outputs into a final answer."""
        successful = [r for r in results if r.success and r.output]
        if not successful:
            return "All agents failed to produce output."

        if len(successful) == 1:
            return successful[0].output

        # Build synthesis prompt
        agent_outputs = "\n\n".join(
            f"### {r.agent_name} ({r.role})\n{r.output}"
            for r in successful
        )
        synthesis_prompt = (
            f"<|system|>\nYou are a Senior Coordinator synthesizing expert analyses. "
            f"Your job: extract the BEST insights from each expert, resolve conflicts, "
            f"and produce a single coherent, actionable answer. "
            f"Do NOT just summarize — synthesize. Find consensus where it exists, "
            f"flag genuine disagreements.\n"
            f"\n<|user|>\nOriginal task: {task}\n\n"
            f"Expert analyses ({pattern} pattern):\n{agent_outputs}\n\n"
            f"Synthesize the above into a comprehensive, actionable response.\n"
            f"<|assistant|>\n"
        )

        try:
            if self._model:
                return self._model.generate(synthesis_prompt, max_tokens=2000) or agent_outputs
        except Exception:
            pass

        # Fallback: concatenate
        return agent_outputs

    # ─────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────
    def _make_result(
        self, task: str, pattern: str, results: list[AgentResult],
        synthesis: str, total_ms: float,
    ) -> SwarmResult:
        sr = SwarmResult(
            task=task,
            pattern=pattern,
            agent_results=results,
            synthesis=synthesis,
            total_time_ms=total_ms,
            success_count=sum(1 for r in results if r.success),
            failure_count=sum(1 for r in results if not r.success),
        )
        self._results_history.append(sr)

        # Track with learning engine
        if get_learning_engine:
            try:
                le = get_learning_engine()
                le.capture(
                    user_input=f"[SWARM:{pattern}] {task}",
                    assistant_response=synthesis,
                    tools_used=[f"swarm_{pattern.lower()}"],
                    response_time_ms=total_ms,
                    success=sr.success_count > 0,
                )
            except Exception:
                pass

        return sr

    def _empty_result(self, task: str, pattern: str) -> SwarmResult:
        return SwarmResult(
            task=task, pattern=pattern, agent_results=[],
            synthesis="No agents available for this swarm.",
            total_time_ms=0, success_count=0, failure_count=0,
        )

    def get_available_agents(self) -> list[dict]:
        return [{"key": k, "name": a.name, "role": a.role} for k, a in self._agents.items()]

    def get_history_summary(self) -> list[dict]:
        return [
            {
                "task": r.task[:80],
                "pattern": r.pattern,
                "agents": r.success_count,
                "time_ms": round(r.total_time_ms),
            }
            for r in self._results_history[-20:]
        ]
