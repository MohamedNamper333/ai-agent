import ast
import os
import re
from pathlib import Path
from collections import Counter


class CodeAnalysis:
    @staticmethod
    def scan_project(path: str = "", max_files: int = 50) -> str:
        root = Path(path) if path else Path.cwd()
        if not root.exists():
            return f"Error: Path not found: {path}"

        files = []
        dirs = []
        for entry in sorted(root.rglob("*")):
            if entry.is_dir():
                if not entry.name.startswith(".") and entry.name != "__pycache__":
                    dirs.append(entry)
            elif entry.is_file():
                files.append(entry)

        ext_count = Counter()
        for f in files:
            ext_count[f.suffix.lower()] += 1

        lines = [
            f"Project: {root.absolute()}",
            f"Total files: {len(files)}",
            f"Total dirs: {len(dirs)}",
            "",
            "File types:",
        ]
        for ext, count in ext_count.most_common(20):
            lines.append(f"  {ext or '(no ext)'}: {count}")

        lines.append("")
        lines.append("Structure (top 3 levels):")
        prefix = ""
        for d in sorted(dirs):
            try:
                rel = d.relative_to(root)
                depth = len(rel.parts)
                if depth <= 3:
                    lines.append(f"  {'  ' * depth}{d.name}/")
            except ValueError:
                continue

        return "\n".join(lines)

    @staticmethod
    def review_code(file_path: str) -> str:
        p = Path(file_path)
        if not p.exists():
            return f"Error: File not found: {file_path}"

        try:
            source = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"Error reading file: {e}"

        ext = p.suffix.lower()
        lines = source.splitlines()
        issues = []
        stats = {
            "lines": len(lines),
            "code_lines": 0,
            "comment_lines": 0,
            "blank_lines": 0,
            "functions": 0,
            "classes": 0,
        }

        for line in lines:
            stripped = line.strip()
            if not stripped:
                stats["blank_lines"] += 1
            elif stripped.startswith("#") or stripped.startswith("//") or stripped.startswith("/*"):
                stats["comment_lines"] += 1
            else:
                stats["code_lines"] += 1

        if ext == ".py":
            try:
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        stats["functions"] += 1
                    elif isinstance(node, ast.ClassDef):
                        stats["classes"] += 1
            except SyntaxError as e:
                issues.append(f"Python syntax error: {e}")

            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                if len(line) > 100:
                    issues.append(f"Line {i}: too long ({len(line)} chars, max 100)")
                if stripped and not stripped.startswith("#"):
                    if re.search(r'(?<!\w)print\(', stripped) and 'logger' not in source:
                        issues.append(f"Line {i}: raw print() detected - use logging?")
                    if re.search(r'(?<!\w)eval\(', stripped):
                        issues.append(f"Line {i}: eval() detected - security risk")
                    if re.search(r'(?<!\w)exec\(', stripped):
                        issues.append(f"Line {i}: exec() detected - security risk")
                    if "except:" in stripped and "Exception" not in stripped:
                        issues.append(f"Line {i}: bare except - too broad")
                    if "TODO" in stripped or "FIXME" in stripped:
                        issues.append(f"Line {i}: TODO/FIXME found")

        report = [
            f"Code Review: {file_path}",
            "",
            "Statistics:",
            f"  Total lines: {stats['lines']}",
            f"  Code: {stats['code_lines']}",
            f"  Comments: {stats['comment_lines']}",
            f"  Blank: {stats['blank_lines']}",
            f"  Functions: {stats['functions']}",
            f"  Classes: {stats['classes']}",
        ]

        if issues:
            unique_issues = list(dict.fromkeys(issues))
            report.append("")
            report.append(f"Issues found ({len(unique_issues)}):")
            for issue in unique_issues[:20]:
                report.append(f"  ⚠ {issue}")
            if len(unique_issues) > 20:
                report.append(f"  ... and {len(unique_issues) - 20} more")
        else:
            report.append("")
            report.append("No issues found - code looks clean!")

        return "\n".join(report)

    @staticmethod
    def analyze_imports(file_path: str) -> str:
        p = Path(file_path)
        if not p.exists():
            return f"Error: File not found: {file_path}"

        try:
            source = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"Error reading file: {e}"

        imports = []
        stdlib = {"os", "sys", "json", "re", "math", "datetime", "pathlib", "collections",
                   "typing", "functools", "itertools", "hashlib", "subprocess", "time",
                   "random", "ast", "argparse", "copy", "enum", "io", "textwrap", "uuid"}

        for line in source.splitlines():
            stripped = line.strip()
            m = re.match(r'^import\s+(\S+)', stripped)
            if m:
                name = m.group(1).split(".")[0]
                category = "stdlib" if name in stdlib else "third-party" if name != "config" else "local"
                imports.append((name, category))

            m = re.match(r'^from\s+(\S+)\s+import', stripped)
            if m:
                name = m.group(1).split(".")[0]
                category = "stdlib" if name in stdlib else "third-party" if name != "config" else "local"
                imports.append((name, category))

        if not imports:
            return f"No imports found in {file_path}"

        by_cat = {"stdlib": [], "third-party": [], "local": []}
        for name, cat in imports:
            if name not in by_cat[cat]:
                by_cat[cat].append(name)

        lines = [f"Imports in {file_path}:", ""]
        for cat, label in [("stdlib", "Standard Library"), ("third-party", "Third Party"), ("local", "Local")]:
            if by_cat[cat]:
                lines.append(f"  {label}:")
                for name in sorted(by_cat[cat]):
                    lines.append(f"    - {name}")
                lines.append("")

        return "\n".join(lines)
