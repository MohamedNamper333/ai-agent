"""Self-Improvement tools - analyze, review, and improve own codebase"""

import ast
import re
import tempfile
import time
from pathlib import Path
from typing import Optional

from core.model import LLM


class SelfImprover:
    def __init__(self, model: Optional[LLM] = None):
        self.model = model or LLM()
        self.project_dir = Path(__file__).parent.parent

    def analyze_codebase(self) -> str:
        files = []
        total_lines = 0
        total_files = 0

        for f in sorted(self.project_dir.rglob("*.py")):
            if "venv" in str(f) or "__pycache__" in str(f) or ".git" in str(f):
                continue
            rel = f.relative_to(self.project_dir)
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
                lines = len(content.splitlines())
                files.append(f"{rel} ({lines} lines)")
                total_lines += lines
                total_files += 1
            except Exception:
                continue

        categories = {}
        for f in files:
            parts = f.split("/")
            if len(parts) > 1:
                cat = parts[0]
            else:
                cat = "root"
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(f)

        result = [
            f"Codebase: {self.project_dir.name}",
            f"Total files: {total_files}",
            f"Total lines: {total_lines}",
            "",
        ]

        for cat, cat_files in sorted(categories.items()):
            result.append(f"### {cat}/ ({len(cat_files)} files)")
            for f in cat_files:
                result.append(f"  {f}")
            result.append("")

        return "\n".join(result)

    def suggest_improvements(self, file_path: str) -> str:
        p = Path(file_path)
        if not p.exists() or not p.is_file():
            return f"File not found: {file_path}"

        code = p.read_text(encoding="utf-8", errors="replace")

        issues = self._analyze_code_issues(code, p.suffix)

        prompt = (
            f"<|system|>\nYou are a code improvement expert. Review this Python code "
            f"and suggest specific, actionable improvements.\n"
            f"Focus on: performance, readability, error handling, security, bugs.\n"
            f"For each suggestion, provide the exact code change.\n"
            f"Here are issues found by static analysis:\n{chr(10).join(issues)}\n"
            f"<|user|>\n```python\n{code}\n```\n<|assistant|>\n"
        )
        return self.model.generate(prompt, max_tokens=1500)

    def apply_improvement(self, file_path: str, instructions: str) -> str:
        p = Path(file_path)
        if not p.exists() or not p.is_file():
            return f"File not found: {file_path}"

        original = p.read_text(encoding="utf-8", errors="replace")

        prompt = (
            f"<|system|>\nYou are a code editor. Apply the requested improvement."
            f"\nReturn ONLY the complete modified code in a triple-backtick python block.\n"
            f"<|user|>\n## File: {file_path}\n\n## Code:\n```python\n{original}\n```\n\n"
            f"## Request:\n{instructions}\n<|assistant|>\n"
        )

        response = self.model.generate(prompt, max_tokens=3000)
        code_match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
        if not code_match:
            return f"Could not extract code from response:\n{response[:500]}"

        new_code = code_match.group(1)

        try:
            ast.parse(new_code)
        except SyntaxError as e:
            return f"Generated code has syntax errors: {e}. Not applied."

        import shutil
        backup = str(p) + f".bak.{int(time.time())}"
        shutil.copy2(str(p), backup)

        try:
            p.write_text(new_code, encoding="utf-8")
            return f"Applied improvement. Backup: {backup}"
        except Exception as e:
            shutil.copy2(backup, str(p))
            return f"Error applying improvement: {e}. Original restored."

    def run_self_review(self) -> str:
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

    def get_code_metrics(self) -> str:
        metrics = {
            "total_files": 0,
            "total_lines": 0,
            "total_functions": 0,
            "total_classes": 0,
            "total_comments": 0,
            "avg_function_length": 0,
            "avg_class_length": 0,
        }

        function_lengths = []
        class_lengths = []

        for f in sorted(self.project_dir.rglob("*.py")):
            if "venv" in str(f) or "__pycache__" in str(f) or ".git" in str(f):
                continue

            try:
                content = f.read_text(encoding="utf-8", errors="replace")
                lines = content.splitlines()
                metrics["total_files"] += 1
                metrics["total_lines"] += len(lines)

                for line in lines:
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        metrics["total_comments"] += 1

                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        metrics["total_functions"] += 1
                        func_lines = node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 0
                        function_lengths.append(func_lines)
                    elif isinstance(node, ast.ClassDef):
                        metrics["total_classes"] += 1
                        class_lines = node.end_lineno - node.lineno + 1 if hasattr(node, 'end_lineno') else 0
                        class_lengths.append(class_lines)
            except Exception:
                continue

        if function_lengths:
            metrics["avg_function_length"] = sum(function_lengths) / len(function_lengths)
        if class_lengths:
            metrics["avg_class_length"] = sum(class_lengths) / len(class_lengths)

        result = [
            "Codebase Metrics:",
            f"  Files: {metrics['total_files']}",
            f"  Lines: {metrics['total_lines']}",
            f"  Functions: {metrics['total_functions']}",
            f"  Classes: {metrics['total_classes']}",
            f"  Comment lines: {metrics['total_comments']}",
            f"  Avg function length: {metrics['avg_function_length']:.1f} lines",
            f"  Avg class length: {metrics['avg_class_length']:.1f} lines",
        ]

        return "\n".join(result)

    def _analyze_code_issues(self, code: str, suffix: str) -> list[str]:
        issues = []
        lines = code.splitlines()

        for i, line in enumerate(lines, 1):
            stripped = line.strip()

            if len(line) > 120:
                issues.append(f"Line {i}: Line too long ({len(line)} chars)")

            if stripped and not stripped.startswith("#"):
                if re.search(r'(?<!\w)print\(', stripped):
                    issues.append(f"Line {i}: Use logging instead of print()")

                if re.search(r'(?<!\w)eval\(', stripped):
                    issues.append(f"Line {i}: eval() is a security risk")

                if re.search(r'except\s*:', stripped):
                    issues.append(f"Line {i}: Bare except clause")

        if suffix == ".py":
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if len(node.args.args) > 5:
                            issues.append(f"Function '{node.name}' has {len(node.args.args)} parameters (consider reducing)")
            except SyntaxError:
                issues.append("Syntax error in file")

        return issues
