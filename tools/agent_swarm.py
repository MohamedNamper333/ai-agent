"""tools/agent_swarm.py — Agent Swarm v2 (Kimi-class Intelligence)

Upgrades over v1:
  ✦ Hierarchical swarm: Master → Sub-swarms → Workers
  ✦ Dynamic task decomposition (recursive subtask splitting)
  ✦ Inter-agent memory sharing (ContextBus)
  ✦ Confidence scoring + uncertainty propagation
  ✦ Self-correction loops (agent challenges its own output)
  ✦ Citation tracking ([CITE: source])
  ✦ Adaptive routing via task fingerprinting
  ✦ 12 specialist agents (was 8): + red_team, ethicist, integrator, forecaster
  ✦ ADVERSARIAL pattern: build → red-team → defend
  ✦ RECURSIVE pattern: decompose → sub-swarms → integrate
"""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Optional


class ContextBus:
    """Thread-safe shared memory for inter-agent communication."""

    def __init__(self):
        self._store: dict[str, Any] = {}
        self._timeline: list[dict] = []

    def write(self, agent: str, key: str, value: Any) -> None:
        self._store[key] = value
        self._timeline.append({"agent": agent, "key": key, "ts": time.time()})

    def read(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def summary(self) -> str:
        lines = [f"[{e['agent']}] shared: {e['key']}" for e in self._timeline[-8:]]
        return "\n".join(lines)


@dataclass
class AgentResult:
    agent_name: str
    role: str
    output: str
    success: bool
    time_ms: float
    confidence: float = 0.8
    citations: list = field(default_factory=list)
    error: str = ""
    pass_number: int = 1

    def summary(self) -> str:
        s = "✓" if self.success else "✗"
        return f"[{s} {self.agent_name} | {self.time_ms:.0f}ms | {self.confidence:.0%}]\n{self.output}"


@dataclass
class TaskNode:
    task_id: str
    description: str
    parent_id: Optional[str]
    subtasks: list = field(default_factory=list)
    result: Optional[AgentResult] = None
    depth: int = 0

    @property
    def is_leaf(self) -> bool:
        return not self.subtasks


@dataclass
class SwarmResult:
    task: str
    pattern: str
    agent_results: list
    synthesis: str
    total_time_ms: float
    success_count: int
    failure_count: int
    avg_confidence: float = 0.0
    task_tree: Optional[TaskNode] = None

    def to_text(self) -> str:
        return self.synthesis

    def to_detailed(self) -> str:
        lines = [
            f"## Swarm [{self.pattern}] | agents={self.success_count} | "
            f"confidence={self.avg_confidence:.0%} | {self.total_time_ms:.0f}ms",
            f"Task: {self.task[:120]}",
            "---",
            "## Synthesis",
            self.synthesis,
            "---",
            f"## Agent Outputs ({len(self.agent_results)})",
        ]
        for r in self.agent_results:
            lines.append(r.summary())
        return "\n\n".join(lines)


@dataclass
class SwarmAgent:
    name: str
    role: str
    system_prompt: str
    temperature: float = 0.7
    max_tokens: int = 2000
    timeout_s: int = 90
    priority: int = 1
    enables_self_correction: bool = True
    _model: Any = field(default=None, repr=False)

    def run(self, task: str, context: str = "", bus: Optional[ContextBus] = None, pass_number: int = 1) -> AgentResult:
        start = time.time()
        prompt = self._build_prompt(task, context, bus)
        try:
            raw = self._call(prompt)
            elapsed = (time.time() - start) * 1000
            if self.enables_self_correction and raw and len(raw) > 100 and pass_number == 1:
                corrected = self._self_correct(task, raw)
                if corrected and corrected != raw:
                    raw = corrected
            confidence = self._estimate_confidence(raw)
            if bus and raw:
                bus.write(self.name, f"{self.name}_output", raw[:400])
            import re
            citations = re.findall(r'\[CITE:\s*([^\]]+)\]', raw)
            return AgentResult(
                agent_name=self.name, role=self.role, output=raw or "",
                success=bool(raw), time_ms=elapsed,
                confidence=confidence, citations=citations, pass_number=pass_number,
            )
        except Exception as exc:
            elapsed = (time.time() - start) * 1000
            return AgentResult(
                agent_name=self.name, role=self.role, output="",
                success=False, time_ms=elapsed, error=str(exc), pass_number=pass_number,
            )

    def _build_prompt(self, task: str, context: str, bus: Optional[ContextBus]) -> str:
        parts = [f"<|system|>\n{self.system_prompt}"]
        parts.append("\nState confidence as CONFIDENCE: X%. Mark facts as [CITE: source]. No filler.")
        if bus:
            s = bus.summary()
            if s:
                parts.append(f"\n## Shared context\n{s}")
        if context:
            parts.append(f"\n## Context\n{context}")
        parts.append(f"\n<|user|>\n{task}\n<|assistant|>\n")
        return "\n".join(parts)

    def _self_correct(self, task: str, draft: str) -> str:
        prompt = (
            f"<|system|>Review your own work.\n"
            f"<|user|>Task: {task}\nDraft:\n{draft}\n\n"
            f"Find errors or gaps. If correct, respond EXACTLY: APPROVED\n"
            f"Otherwise respond with corrected version only.\n<|assistant|>\n"
        )
        try:
            result = self._call(prompt, max_tokens=800)
            if result and "APPROVED" not in result.upper()[:20]:
                return result
        except Exception:
            pass
        return draft

    def _estimate_confidence(self, output: str) -> float:
        if not output:
            return 0.0
        import re
        m = re.search(r'CONFIDENCE:\s*(\d+)%', output, re.IGNORECASE)
        if m:
            return int(m.group(1)) / 100
        base = min(0.9, len(output) / 2000 * 0.5 + 0.4)
        certain = sum(output.lower().count(w) for w in ["clearly", "confirmed", "proven"])
        uncertain = sum(output.lower().count(w) for w in ["might", "possibly", "unclear"])
        return min(1.0, max(0.1, base + certain * 0.02 - uncertain * 0.03))

    def _call(self, prompt: str, max_tokens: int = 0) -> str:
        mt = max_tokens or self.max_tokens
        try:
            from core.llm import LLMRouter, LLMRequest, ReasoningLevel
            req = LLMRequest(prompt=prompt, max_tokens=mt, level=ReasoningLevel.DEEP)
            return LLMRouter().generate(req).text or ""
        except Exception:
            pass
        try:
            if self._model:
                return self._model.generate(prompt, max_tokens=mt) or ""
        except Exception:
            pass
        return ""


BUILTIN_AGENTS: dict[str, dict] = {
    "analyst": {
        "name": "The Analyst", "role": "Root-cause analysis",
        "system_prompt": "You are The Analyst. Find root causes, not symptoms. Quantify every finding. Format: numbered findings with evidence + impact 1-10.",
        "temperature": 0.2,
    },
    "architect": {
        "name": "The Architect", "role": "System design",
        "system_prompt": "You are The Architect. Think in trade-offs: cost vs scalability, speed vs correctness. Format: ADR with context, decision, consequences.",
        "temperature": 0.3,
    },
    "security": {
        "name": "The Security Expert", "role": "Threat modeling",
        "system_prompt": "You are The Security Expert. Think like an attacker. OWASP Top 10, supply chain, auth flows. Format: threats with CVSS severity + mitigation.",
        "temperature": 0.1,
    },
    "optimizer": {
        "name": "The Optimizer", "role": "Performance and efficiency",
        "system_prompt": "You are The Optimizer. Big-O, cache hits, batch ops. Measure before optimizing. Format: opportunities ranked by impact % + implementation cost.",
        "temperature": 0.2,
    },
    "critic": {
        "name": "The Critic", "role": "Devil's advocate",
        "system_prompt": "You are The Critic. Disagree and find what others miss. Challenge assumptions. Format: challenges ranked by probability x impact.",
        "temperature": 0.9,
    },
    "researcher": {
        "name": "The Researcher", "role": "Knowledge synthesis",
        "system_prompt": "You are The Researcher. Connect ideas across domains. Cite established patterns. Format: recommendations with pattern names and evidence.",
        "temperature": 0.4,
    },
    "coder": {
        "name": "The Coder", "role": "Code implementation",
        "system_prompt": "You are The Coder. SOLID, DRY, testability, error handling. Production-ready only. No TODOs. Format: working code with comments only where non-obvious.",
        "temperature": 0.1,
    },
    "strategist": {
        "name": "The Strategist", "role": "Business strategy",
        "system_prompt": "You are The Strategist. 18-month horizon. Market forces, user needs, competitive moats. Format: options matrix with trade-offs + recommended path.",
        "temperature": 0.5,
    },
    "red_team": {
        "name": "The Red Team", "role": "Adversarial attacker",
        "system_prompt": "You are The Red Team. ONLY job: break the solution. Edge cases, race conditions, adversarial inputs, misuse. Format: attack vectors with likelihood + exploitability.",
        "temperature": 0.95,
        "enables_self_correction": False,
    },
    "ethicist": {
        "name": "The Ethicist", "role": "Ethics and societal impact",
        "system_prompt": "You are The Ethicist. Bias, fairness, privacy, misuse, 2nd-order effects. Format: concerns with severity + concrete mitigation.",
        "temperature": 0.6,
    },
    "integrator": {
        "name": "The Integrator", "role": "Synthesis and coherence",
        "system_prompt": "You are The Integrator. Combine expert opinions, resolve contradictions by reasoning through them. Cite which expert you agree with. Format: integrated analysis with explicit conflict resolutions.",
        "temperature": 0.3,
    },
    "forecaster": {
        "name": "The Forecaster", "role": "Consequence prediction",
        "system_prompt": "You are The Forecaster. Probabilistic thinking, 3 scenarios: optimistic/realistic/pessimistic with probabilities. Identify key uncertainties. Format: scenario matrix with probabilities + leading indicators.",
        "temperature": 0.7,
    },
}


class TaskDecomposer:
    def __init__(self, model=None):
        self._model = model

    def decompose(self, task: str, max_depth: int = 2) -> TaskNode:
        root = TaskNode(task_id=self._id(task), description=task, parent_id=None, depth=0)
        if max_depth > 0 and self._is_complex(task):
            for st in self._split(task):
                root.subtasks.append(TaskNode(task_id=self._id(st), description=st, parent_id=root.task_id, depth=1))
        return root

    def _is_complex(self, task: str) -> bool:
        return len(task) > 200 or task.count("\n") > 3 or any(w in task.lower() for w in ["and also", "additionally", "design and implement", "analyze and fix"])

    def _split(self, task: str) -> list:
        if not self._model:
            return []
        prompt = (
            f"Break this task into 2-4 independent subtasks.\n"
            f"Task: {task}\n\n"
            f"Respond ONLY with a JSON array of strings. No explanation."
        )
        try:
            raw = self._call(prompt)
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()
            result = json.loads(raw.strip())
            return [s for s in result if isinstance(s, str)][:4]
        except Exception:
            return []

    def _call(self, prompt: str) -> str:
        try:
            from core.llm import LLMRouter, LLMRequest, ReasoningLevel
            req = LLMRequest(prompt=prompt, max_tokens=300, level=ReasoningLevel.SIMPLE)
            return LLMRouter().generate(req).text or ""
        except Exception:
            pass
        try:
            if self._model:
                return self._model.generate(prompt, max_tokens=300) or ""
        except Exception:
            pass
        return ""

    @staticmethod
    def _id(text: str) -> str:
        return hashlib.md5(text.encode()).hexdigest()[:8]


class AgentSwarm:
    """
    Agent Swarm v2 — Kimi-class intelligence.

    Patterns:
      PARALLEL    — all agents simultaneously (fastest)
      PIPELINE    — sequential chain (most coherent)
      DEBATE      — multi-round argumentation (most accurate)
      ADVERSARIAL — build + red-team attack + defense (most rigorous)
      RECURSIVE   — decompose into sub-swarms (handles complex tasks)
      AUTO        — fingerprint-based selection
    """

    def __init__(self, model=None, max_workers: int = 6):
        self._model = model
        self._max_workers = max_workers
        self._agents: dict[str, SwarmAgent] = {}
        self._bus = ContextBus()
        self._decomposer = TaskDecomposer(model)
        self._history: list[SwarmResult] = []
        self._load_agents()

    def _load_agents(self) -> None:
        for key, cfg in BUILTIN_AGENTS.items():
            self._agents[key] = SwarmAgent(
                name=cfg["name"], role=cfg["role"],
                system_prompt=cfg["system_prompt"],
                temperature=cfg.get("temperature", 0.7),
                enables_self_correction=cfg.get("enables_self_correction", True),
                _model=self._model,
            )

    def add_agent(self, key: str, agent: SwarmAgent) -> None:
        self._agents[key] = agent

    def run_parallel(self, task: str, agent_keys: Optional[list] = None, timeout_s: int = 120) -> SwarmResult:
        self._bus = ContextBus()
        keys = agent_keys or list(self._agents.keys())
        agents = [self._agents[k] for k in keys if k in self._agents]
        start = time.time()
        results = self._run_parallel(task, agents, timeout_s=timeout_s)
        synthesis = self._synthesize(task, results, "PARALLEL")
        return self._make_result(task, "PARALLEL", results, synthesis, time.time() - start)

    def run_pipeline(self, task: str, agent_keys: Optional[list] = None) -> SwarmResult:
        self._bus = ContextBus()
        keys = agent_keys or ["researcher", "architect", "coder", "security", "critic"]
        agents = [self._agents[k] for k in keys if k in self._agents]
        start = time.time()
        results: list[AgentResult] = []
        ctx = ""
        for agent in agents:
            r = agent.run(task, context=ctx, bus=self._bus)
            results.append(r)
            if r.success:
                ctx += f"\n\n### {agent.name}\n{r.output}"
        synthesis = self._synthesize(task, results, "PIPELINE")
        return self._make_result(task, "PIPELINE", results, synthesis, time.time() - start)

    def run_debate(self, task: str, agent_keys: Optional[list] = None, rounds: int = 2) -> SwarmResult:
        self._bus = ContextBus()
        keys = agent_keys or ["analyst", "critic", "architect", "optimizer", "forecaster"]
        agents = [self._agents[k] for k in keys if k in self._agents]
        start = time.time()
        all_results: list[AgentResult] = []
        r1 = self._run_parallel(task, agents, pass_number=1)
        all_results.extend(r1)
        if rounds >= 2:
            ctx = "\n\n".join(f"### {r.agent_name}\n{r.output}" for r in r1 if r.success)
            rebuttal = f"Experts said:\n\n{ctx}\n\nRefine your position. Address their arguments.\n\nOriginal: {task}"
            r2 = self._run_parallel(rebuttal, agents, pass_number=2)
            all_results.extend(r2)
        synthesis = self._synthesize(task, all_results, "DEBATE")
        return self._make_result(task, "DEBATE", all_results, synthesis, time.time() - start)

    def run_adversarial(self, task: str, solution_agents: Optional[list] = None) -> SwarmResult:
        self._bus = ContextBus()
        start = time.time()
        all_results: list[AgentResult] = []
        builders = solution_agents or ["architect", "coder", "security"]
        build_agents = [self._agents[k] for k in builders if k in self._agents]
        build_results = self._run_parallel(task, build_agents)
        all_results.extend(build_results)
        solution_ctx = "\n\n".join(f"### {r.agent_name}: {r.output}" for r in build_results if r.success)
        if "red_team" in self._agents:
            attack_task = f"The team proposed:\n\n{solution_ctx}\n\nATTACK IT. Find every weakness."
            rt_result = self._agents["red_team"].run(attack_task, bus=self._bus)
            all_results.append(rt_result)
            defense_ctx = f"Red team attack:\n{rt_result.output}\n\nDefend or improve the solution."
            defense_agents = [self._agents[k] for k in ["analyst", "architect"] if k in self._agents]
            all_results.extend(self._run_parallel(defense_ctx, defense_agents))
        synthesis = self._synthesize(task, all_results, "ADVERSARIAL")
        return self._make_result(task, "ADVERSARIAL", all_results, synthesis, time.time() - start)

    def run_recursive(self, task: str, max_depth: int = 2) -> SwarmResult:
        self._bus = ContextBus()
        start = time.time()
        tree = self._decomposer.decompose(task, max_depth)
        all_results: list[AgentResult] = []
        if tree.subtasks:
            with ThreadPoolExecutor(max_workers=self._max_workers) as ex:
                futures = {ex.submit(self._mini_swarm, node.description): node for node in tree.subtasks}
                for future in as_completed(futures, timeout=180):
                    try:
                        sub = future.result(timeout=10)
                        all_results.extend(sub.agent_results)
                    except Exception as e:
                        all_results.append(AgentResult(agent_name="SubSwarm", role="recursive", output="", success=False, error=str(e), time_ms=0))
        else:
            all_results.extend(self._run_parallel(task, list(self._agents.values())[:6]))
        synthesis = self._synthesize(task, all_results, "RECURSIVE")
        return self._make_result(task, "RECURSIVE", all_results, synthesis, time.time() - start, task_tree=tree)

    def run_auto(self, task: str) -> SwarmResult:
        p = self._fingerprint(task)
        if p == "code":
            return self.run_pipeline(task, ["researcher", "coder", "security", "optimizer", "critic"])
        elif p == "decision":
            return self.run_debate(task)
        elif p == "critical":
            return self.run_adversarial(task)
        elif p == "complex":
            return self.run_recursive(task)
        return self.run_parallel(task)

    def _fingerprint(self, task: str) -> str:
        t = task.lower()
        if any(w in t for w in ["write code", "implement", "fix bug", "create function", "build api"]):
            return "code"
        if any(w in t for w in ["should i", "which is better", "compare", "choose between", "vs "]):
            return "decision"
        if any(w in t for w in ["critical", "production", "security exploit", "vulnerability"]):
            return "critical"
        if len(task) > 300 or task.count("\n") > 5:
            return "complex"
        return "parallel"

    def _run_parallel(self, task: str, agents: list, pass_number: int = 1, timeout_s: int = 120) -> list:
        results = []
        with ThreadPoolExecutor(max_workers=self._max_workers) as ex:
            futures = {ex.submit(a.run, task, "", self._bus, pass_number): a for a in agents}
            for future in as_completed(futures, timeout=timeout_s):
                agent = futures[future]
                try:
                    results.append(future.result(timeout=5))
                except Exception as e:
                    results.append(AgentResult(agent_name=agent.name, role=agent.role, output="", success=False, error=str(e), time_ms=0))
        return results

    def _mini_swarm(self, task: str) -> SwarmResult:
        agents = [self._agents.get(k) for k in ["analyst", "coder", "critic"] if k in self._agents]
        results = self._run_parallel(task, agents)
        synthesis = self._synthesize(task, results, "MINI")
        return SwarmResult(task=task, pattern="MINI", agent_results=results, synthesis=synthesis,
                           total_time_ms=sum(r.time_ms for r in results),
                           success_count=sum(1 for r in results if r.success),
                           failure_count=sum(1 for r in results if not r.success),
                           avg_confidence=self._avg_conf(results))

    def _synthesize(self, task: str, results: list, pattern: str) -> str:
        ok = [r for r in results if r.success and r.output]
        if not ok:
            return "All agents failed."
        if len(ok) == 1:
            return ok[0].output
        if "integrator" in self._agents and len(ok) >= 3:
            ctx = "\n\n".join(f"### {r.agent_name} ({r.confidence:.0%})\n{r.output}" for r in sorted(ok, key=lambda r: r.confidence, reverse=True))
            ir = self._agents["integrator"].run(f"Synthesize. Cite experts. Resolve conflicts.\n\nOriginal: {task}\n\nAnalyses:\n{ctx}", bus=self._bus)
            if ir.success and ir.output:
                return ir.output
        ctx = "\n\n".join(f"### {r.agent_name} ({r.confidence:.0%})\n{r.output}" for r in ok[:8])
        prompt = f"<|system|>Synthesize these expert analyses into one clear actionable response.\nWeight by confidence. Resolve conflicts explicitly.\n<|user|>Task: {task}\n\nExperts:\n{ctx}\n<|assistant|>\n"
        try:
            from core.llm import LLMRouter, LLMRequest, ReasoningLevel
            r = LLMRouter().generate(LLMRequest(prompt=prompt, max_tokens=2000, level=ReasoningLevel.DEEP)).text
            if r and len(r.strip()) > 50:
                return r
        except Exception:
            pass
        try:
            if self._model:
                r = self._model.generate(prompt, max_tokens=2000)
                if r:
                    return r
        except Exception:
            pass
        return ctx

    def _make_result(self, task, pattern, results, synthesis, elapsed_s, task_tree=None):
        sr = SwarmResult(task=task, pattern=pattern, agent_results=results, synthesis=synthesis,
                         total_time_ms=elapsed_s * 1000,
                         success_count=sum(1 for r in results if r.success),
                         failure_count=sum(1 for r in results if not r.success),
                         avg_confidence=self._avg_conf(results), task_tree=task_tree)
        self._history.append(sr)
        try:
            from core.learning_engine import get_learning_engine
            get_learning_engine().capture(user_input=f"[SWARM:{pattern}] {task}", assistant_response=synthesis,
                                          tools_used=["agent_swarm"], response_time_ms=elapsed_s * 1000, success=bool(synthesis))
        except Exception:
            pass
        return sr

    def _avg_conf(self, results: list) -> float:
        good = [r.confidence for r in results if r.success and r.confidence > 0]
        return sum(good) / len(good) if good else 0.0

    def get_available_agents(self) -> list:
        return [{"key": k, "name": a.name, "role": a.role} for k, a in self._agents.items()]

    def get_history(self, n: int = 20) -> list:
        return [{"task": r.task[:80], "pattern": r.pattern, "agents": r.success_count,
                 "confidence": f"{r.avg_confidence:.0%}", "time_ms": round(r.total_time_ms)} for r in self._history[-n:]]

    def get_stats(self) -> dict:
        if not self._history:
            return {"total_runs": 0}
        avg_conf = sum(r.avg_confidence for r in self._history) / len(self._history)
        by_pattern: dict[str, int] = {}
        for r in self._history:
            by_pattern[r.pattern] = by_pattern.get(r.pattern, 0) + 1
        return {"total_runs": len(self._history), "avg_confidence": f"{avg_conf:.0%}",
                "by_pattern": by_pattern, "total_agents": len(self._agents)}
