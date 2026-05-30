"""Multi-Agent System - specialized agents collaborating"""

import json
import re
from typing import Optional

from core.model import LLM


class SpecialistAgent:
    def __init__(self, name: str, role: str, expertise: str, model: Optional[LLM] = None):
        self.name = name
        self.role = role
        self.expertise = expertise
        self.model = model or LLM()

    def process(self, task: str, context: str = "") -> str:
        prompt = (
            f"<|system|>\nYou are {self.name}, a specialist agent.\n"
            f"Role: {self.role}\n"
            f"Expertise: {self.expertise}\n\n"
            f"Respond with your expert analysis of the following task.\n"
        )
        if context:
            prompt += f"Context:\n{context}\n\n"
        prompt += f"<|user|>\n{task}\n<|assistant|>\n"
        return self.model.generate(prompt)


class MultiAgentOrchestrator:
    def __init__(self, model: Optional[LLM] = None):
        self.model = model or LLM()
        self.specialists = [
            SpecialistAgent(
                "The Analyst",
                "Data & code analyst",
                "Analyzing data, finding patterns, debugging code, reviewing logic",
                self.model,
            ),
            SpecialistAgent(
                "The Programmer",
                "Software developer",
                "Writing clean code, designing architecture, implementing features",
                self.model,
            ),
            SpecialistAgent(
                "The Reviewer",
                "Code reviewer & QA",
                "Code review, bug detection, security audit, best practices",
                self.model,
            ),
        ]

    def run_council(self, task: str, context: str = "") -> str:
        results = []
        for agent in self.specialists:
            response = agent.process(task, context)
            results.append({"agent": agent.name, "response": response})

        combined = "\n\n".join(
            f"### {r['agent']} says:\n{r['response']}" for r in results
        )

        synthesis_prompt = (
            f"<|system|>\nYou are a synthesis coordinator. Multiple specialist agents "
            f"have analyzed a task. Combine their insights into a coherent final response.\n"
            f"<|user|>\n## Task\n{task}\n\n## Specialist Responses\n{combined}\n\n"
            f"## Your Synthesis\nProvide a unified response that incorporates the best "
            f"insights from all specialists. Note any disagreements.\n"
            f"<|assistant|>\n"
        )
        synthesis = self.model.generate(synthesis_prompt)
        return synthesis

    def delegate(self, agent_name: str, task: str, context: str = "") -> str:
        for agent in self.specialists:
            if agent.name.lower() == agent_name.lower():
                return agent.process(task, context)
        return f"Unknown specialist: {agent_name}"
