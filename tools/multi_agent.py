"""Multi-Agent System - specialized agents collaborating with ensemble reasoning"""

import json
import re
import time
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.model import LLM


class SpecialistAgent:
    def __init__(self, name: str, role: str, expertise: str, instructions: str,
                 model: Optional[LLM] = None):
        self.name = name
        self.role = role
        self.expertise = expertise
        self.instructions = instructions
        self.model = model or LLM()
        self.memory: list[dict] = []

    def process(self, task: str, context: str = "", tools: str = "",
                previous_analyses: str = "") -> str:
        prompt_parts = [
            f"<|system|>\nYou are {self.name}, a specialist AI agent.",
            f"Role: {self.role}",
            f"Expertise: {self.expertise}",
            f"\n## Operating Instructions",
            self.instructions,
        ]

        if previous_analyses:
            prompt_parts.extend([
                f"\n## Previous Analyses",
                f"Other specialists have already analyzed this. Build on their work:",
                previous_analyses,
            ])

        if tools:
            prompt_parts.append(f"\n## Available Tools\n{tools}")

        if context:
            prompt_parts.append(f"\n## Context\n{context}")

        prompt_parts.extend([
            f"\n## Your Task",
            f"Provide your expert analysis of the following. Be specific, actionable, and thorough.",
            f"Use your expertise to provide insights that other specialists might miss.",
            f"<|user|>\n{task}\n<|assistant|>\n",
        ])

        response = self.model.generate("\n".join(prompt_parts), max_tokens=2000)

        self.memory.append({
            "task": task[:100],
            "response": response[:500],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

        return response

    def get_history(self) -> list[dict]:
        return list(self.memory)


class MultiAgentOrchestrator:
    def __init__(self, model: Optional[LLM] = None):
        self.model = model or LLM()

        self.specialists = [
            SpecialistAgent(
                "The Analyst",
                "Data & code analyst",
                "Deep analysis of data, code, and systems. Finding patterns, anomalies, and insights. "
                "Statistical analysis, code review, debugging, performance profiling.",
                "1. Start by understanding what the data/code is doing\n"
                "2. Look for patterns, anomalies, bottlenecks\n"
                "3. Quantify findings with metrics\n"
                "4. Provide actionable recommendations\n"
                "5. Flag any security concerns",
                self.model,
            ),
            SpecialistAgent(
                "The Programmer",
                "Senior software engineer",
                "Writing clean, efficient, maintainable code. System architecture, API design, "
                "refactoring, optimization. Best practices in Python, design patterns, testing.",
                "1. Design pragmatic, testable solutions\n"
                "2. Follow SOLID principles and clean code\n"
                "3. Consider edge cases and error handling\n"
                "4. Include type hints and documentation\n"
                "5. Suggest benchmark/comparison if relevant",
                self.model,
            ),
            SpecialistAgent(
                "The Reviewer",
                "Code reviewer & QA engineer",
                "Code review, bug detection, security audit, test coverage, performance review, "
                "adherence to best practices, documentation quality.",
                "1. Check for bugs, race conditions, memory leaks\n"
                "2. Review error handling and edge cases\n"
                "3. Check test coverage quality\n"
                "4. Assess security posture\n"
                "5. Verify documentation completeness",
                self.model,
            ),
            SpecialistAgent(
                "The Architect",
                "System architect & designer",
                "Software architecture, system design, scalability, trade-off analysis, "
                "design patterns, technology selection, API design, data modeling.",
                "1. Understand the system's goals and constraints\n"
                "2. Evaluate architecture decisions and trade-offs\n"
                "3. Suggest improvements for scalability and maintainability\n"
                "4. Consider separation of concerns and modularity\n"
                "5. Provide migration path for improvements",
                self.model,
            ),
        ]

    def run_council(self, task: str, context: str = "", parallel: bool = True) -> str:
        if parallel:
            results = self._run_parallel(task, context)
        else:
            results = self._run_sequential(task, context)

        combined = "\n\n".join(
            f"## {r['agent']}\n{r['response']}" for r in results
        )

        synthesis_prompt = (
            f"<|system|>\nYou are a synthesis coordinator. Multiple specialist agents "
            f"have analyzed a task. Synthesize their insights.\n\n"
            f"## Synthesis Guidelines\n"
            f"1. Identify key findings that multiple agents agree on\n"
            f"2. Note any disagreements and explain both perspectives\n"
            f"3. Prioritize findings by impact and urgency\n"
            f"4. Provide a clear, actionable final recommendation\n"
            f"5. Include a confidence assessment for major conclusions\n"
            f"<|user|>\n## Task\n{task}\n\n## Specialist Analyses\n"
            f"{combined}\n\n## Synthesis\n"
            f"<|assistant|>\n"
        )

        synthesis = self.model.generate(synthesis_prompt, max_tokens=2500)
        return synthesis

    def _run_sequential(self, task: str, context: str) -> list[dict]:
        results = []
        previous = ""
        for agent in self.specialists:
            response = agent.process(task, context, previous_analyses=previous)
            results.append({"agent": agent.name, "response": response})
            previous += f"\n### {agent.name}\n{response[:300]}\n"
        return results

    def _run_parallel(self, task: str, context: str) -> list[dict]:
        results = []
        with ThreadPoolExecutor(max_workers=len(self.specialists)) as executor:
            future_to_agent = {
                executor.submit(agent.process, task, context): agent
                for agent in self.specialists
            }
            for future in as_completed(future_to_agent):
                agent = future_to_agent[future]
                try:
                    response = future.result()
                    results.append({"agent": agent.name, "response": response})
                except Exception as e:
                    results.append({
                        "agent": agent.name,
                        "response": f"Error: {e}",
                    })
        return results

    def delegate(self, agent_name: str, task: str, context: str = "", tools: str = "") -> str:
        for agent in self.specialists:
            if agent.name.lower() == agent_name.lower():
                return agent.process(task, context, tools)
        return f"Unknown specialist: {agent_name}. Available: {', '.join(a.name for a in self.specialists)}"

    def debate(self, topic: str, rounds: int = 2) -> str:
        debate_history = []
        for round_num in range(rounds):
            for agent in self.specialists:
                other_views = [
                    f"{h['agent']}: {h['response'][:300]}"
                    for h in debate_history
                    if h['agent'] != agent.name
                ]
                context = ""
                if other_views:
                    context = "Other agents have said:\n" + "\n".join(other_views)
                response = agent.process(f"Round {round_num + 1}: {topic}", context)
                debate_history.append({
                    "agent": agent.name,
                    "round": round_num + 1,
                    "response": response,
                })

        final_prompt = (
            f"<|system|>\nYou are a debate moderator. Agents have debated: {topic}\n\n"
            f"## Debate History\n"
        )
        for h in debate_history:
            final_prompt += f"[R{h['round']}] {h['agent']}: {h['response'][:400]}\n\n"

        final_prompt += (
            "## Final Summary\nProvide a summary of the debate, highlighting:\n"
            "1. Points of agreement\n2. Points of disagreement\n"
            "3. The strongest arguments on each side\n4. A final recommendation\n"
            "<|assistant|>\n"
        )
        return self.model.generate(final_prompt, max_tokens=2000)

    def group_consensus(self, task: str, options: list[str]) -> str:
        votes = []
        for agent in self.specialists:
            prompt = (
                f"<|system|>\nYou are {agent.name}. "
                f"Analyze the following options and vote for the best one.\n\n"
                f"Options:\n" + "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options)) +
                f"\n\n<|user|>\n{task}\n\n"
                f"Respond with JSON: {{\"vote\": <option_number>, \"reason\": \"...\", "
                f"\"confidence\": <0.0-1.0>}}\n<|assistant|>\n"
            )
            try:
                response = self.model.generate(prompt, max_tokens=500)
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    vote_data = json.loads(json_match.group())
                    votes.append(vote_data)
            except Exception:
                continue

        weighted_votes = {}
        valid_votes = []
        for v in votes:
            try:
                idx = int(v.get("vote", 0)) - 1
                confidence = float(v.get("confidence", 0.5))
            except (TypeError, ValueError):
                continue
            valid_votes.append(v)
            if 0 <= idx < len(options):
                weighted_votes[idx] = weighted_votes.get(idx, 0) + confidence

        if not valid_votes:
            return "No consensus could be reached (no votes collected)."

        result = [f"## Group Consensus Analysis\n"]
        result.append(f"Task: {task}\n")
        result.append("### Vote Results")
        for i, opt in enumerate(options):
            score = weighted_votes.get(i, 0)
            bar = "#" * int(score * 20 / max(max(weighted_votes.values(), default=1), 0.01))
            result.append(f"  {opt}: {score:.2f} {bar}")

        winner = max(weighted_votes, key=weighted_votes.get) if weighted_votes else 0
        result.append(f"\n### Winner: {options[winner]}")
        result.append(f"\n### Individual Votes")
        for i, v in enumerate(valid_votes):
            result.append(f"  {self.specialists[i].name if i < len(self.specialists) else f'Agent {i}'}: "
                         f"Option {v.get('vote', '?')} ({float(v.get('confidence', 0)):.0%} confidence)")

        return "\n".join(result)

    def get_specialist_info(self) -> list[dict]:
        return [
            {
                "name": s.name,
                "role": s.role,
                "expertise": s.expertise[:80],
                "tasks_completed": len(s.memory),
            }
            for s in self.specialists
        ]
