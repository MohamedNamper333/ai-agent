"""Auto-Improve - agent analyzes and improves its own code"""

import ast
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

from core.model import LLM


class SelfImprover:
    def __init__(self, model: Optional[LLM] = None):
        self.model = model or LLM()
        self.project_dir = Path(__file__).parent.parent

    def analyze_codebase(self) -> str:
        """Scan own codebase and report structure"""
        files = []
        for f in sorted(self.project_dir.rglob("*.py")):
            if "venv" in str(f) or "__pycache__" in str(f) or ".git" in str(f):
                continue
            rel = f.relative_to(self.project_dir)
            lines = len(f.read_text(encoding="utf-8", errors="replace").splitlines())
            files.append(f"{rel} ({lines} lines)")

        return f"Codebase: {self.project_dir.name}\n" + "\n".join(files)

    def suggest_improvements(self, file_path: str) -> str:
        """Analyze a specific file and suggest improvements"""
        p = Path(file_path)
        if not p.exists():
            return f"File not found: {file_path}"

        code = p.read_text(encoding="utf-8", errors="replace")

        prompt = (
            f"<|system|>\nYou are a code improvement expert. Review this Python code "
            f"and suggest specific, actionable improvements. Focus on:\n"
            f"1. Performance issues\n"
            f"2. Code quality/readability\n"
            f"3. Missing error handling\n"
            f"4. Security concerns\n"
            f"5. Potential bugs\n\n"
            f"For each suggestion, provide the exact code change.\n"
            f"<|user|>\n```python\n{code}\n```\n"
            f"<|assistant|>\n"
        )

        return self.model.generate(prompt, max_tokens=1500)

    def apply_improvement(self, file_path: str, instructions: str) -> str:
        """Apply an improvement to a file based on instructions"""
        p = Path(file_path)
        if not p.exists():
            return f"File not found: {file_path}"

        original = p.read_text(encoding="utf-8", errors="replace")

        prompt = (
            f"<|system|>\nYou are a code editor. Apply the requested improvement "
            f"to the code. Return the COMPLETE modified file.\n"
            f"<|user|>\n## File: {file_path}\n\n## Current Code:\n```python\n{original}\n```\n\n"
            f"## Improvement Request:\n{instructions}\n\n"
            f"Return ONLY the complete modified code in a ```python block.\n"
            f"<|assistant|>\n"
        )

        response = self.model.generate(prompt, max_tokens=3000)

        code_match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
        if code_match:
            new_code = code_match.group(1)
            backup = str(p) + ".bak"
            p.rename(backup)
            p.write_text(new_code, encoding="utf-8")
            return f"Applied improvement. Backup saved to: {backup}"
        return f"Could not extract code from response:\n{response[:500]}"

    def run_self_review(self) -> str:
        """Run a full review of the project"""
        parts = [self.analyze_codebase(), ""]
        files = []
        for f in sorted(self.project_dir.rglob("*.py")):
            if "venv" in str(f) or "__pycache__" in str(f) or ".git" in str(f):
                continue
            if f.parent.name == "__pycache__":
                continue
            files.append(f)

        for f in files[:5]:
            rel = f.relative_to(self.project_dir)
            parts.append(f"--- {rel} ---")
            parts.append(self.suggest_improvements(str(f)))
            parts.append("")

        return "\n".join(parts)
