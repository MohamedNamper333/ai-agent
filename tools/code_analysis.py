import ast
import os
import re
import math
import sys
import builtins
from pathlib import Path
from collections import Counter
from typing import Optional


class CodeAnalysis:
    """Advanced code analysis with AST-based deep analysis, security scanning, and quality scoring."""

    SECURITY_PATTERNS = [
        (r'(?<!\w)eval\s*\(', 'eval() usage - code injection risk', 'CRITICAL'),
        (r'(?<!\w)exec\s*\(', 'exec() usage - code injection risk', 'CRITICAL'),
        (r'__import__\s*\(', '__import__() usage - code injection risk', 'CRITICAL'),
        (r'subprocess\.\s*(call|Popen|run|check_call|check_output)\s*\(', 'subprocess usage - command injection risk', 'HIGH'),
        (r'os\.\s*(system|popen|exec|execl|execle|execlp|execlpe)\s*\(', 'os.exec usage - command injection risk', 'HIGH'),
        (r'pickle\.\s*(loads?|Unpickler)\s*\(', 'pickle deserialization - RCE risk', 'CRITICAL'),
        (r'yaml\.\s*load\s*\(', 'yaml.load() without SafeLoader - RCE risk', 'HIGH'),
        (r'(?<!\w)input\s*\(', 'Using input() in Python 2 is dangerous (use raw_input)', 'MEDIUM'),
        (r'sqlite3\.Cursor\.execute\s*\(\s*f["\']', 'f-string in SQL query - SQL injection risk', 'CRITICAL'),
        (r'execute\s*\(\s*f["\']', 'f-string in SQL query - SQL injection risk', 'CRITICAL'),
        (r'\.format\(.*\{\}', 'str.format() - potential format string vuln with user input', 'MEDIUM'),
        (r'%s.*%\(', '% formatting - SQL injection risk with user input', 'HIGH'),
        (r'(?<!\w)marshal\.\s*loads?\s*\(', 'marshal deserialization - RCE risk', 'CRITICAL'),
        (r'shelve\.\s*open\s*\(', 'shelve usage - potential arbitrary code execution', 'MEDIUM'),
        (r'tempfile\.mktemp\s*\(', 'mktemp() is deprecated and insecure - use mkstemp()', 'MEDIUM'),
        (r'assert\s+.*True|assert\s+.*False', 'assert with constant - no-op if optimization enabled', 'LOW'),
        (r'(?<!\w)globals\(\)', 'globals() usage - potential security concern', 'LOW'),
        (r'(?<!\w)locals\(\)', 'locals() usage - potential security concern', 'LOW'),
        (r'flask\.render_template_string\s*\(', 'render_template_string() - SSTI risk with user input', 'CRITICAL'),
        (r'\.__subclasses__\(\)', 'Accessing __subclasses__() - sandbox escape attempt', 'CRITICAL'),
        (r'\.__globals__', 'Accessing __globals__ - sandbox escape attempt', 'CRITICAL'),
        (r'\.__builtins__', 'Accessing __builtins__ - sandbox escape attempt', 'CRITICAL'),
        (r'(?<!\w)breakpoint\s*\(', 'breakpoint() left in production code', 'MEDIUM'),
        (r'(?<!\w)pdb\.set_trace\s*\(', 'pdb.set_trace() left in production code', 'MEDIUM'),
        (r'socket\.\s*connect\s*\(', 'Raw socket connection', 'LOW'),
        (r'(?<!\w)compile\s*\(', 'compile() usage - potential code injection', 'HIGH'),
    ]

    CODE_SMELL_PATTERNS = [
        (r'except\s*:', 'Bare except clause - catches ALL exceptions', 'MEDIUM'),
        (r'except\s+Exception\s*:', 'Broad except clause - consider specific exceptions', 'LOW'),
        (r'(?<!\w)print\(', 'print() in production - use logging instead', 'LOW'),
        (r'TODO', 'TODO found - incomplete implementation', 'INFO'),
        (r'FIXME', 'FIXME found - known bug', 'HIGH'),
        (r'HACK', 'HACK found - fragile code', 'MEDIUM'),
        (r'XXX', 'XXX found - known issue', 'MEDIUM'),
        (r'import \*', 'Wildcard import - namespace pollution', 'MEDIUM'),
        (r'\.__dict__', 'Direct __dict__ access - use getattr/setattr', 'LOW'),
        (r'type\(.*\)\(\)', 'Dynamic class creation - hard to debug', 'LOW'),
        (r'\#\s*type:\s*ignore', 'type: ignore suppression', 'INFO'),
        (r'noqa', 'noqa suppression', 'INFO'),
    ]

    @staticmethod
    def scan_project(path: str = "", max_files: int = 100) -> str:
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

        py_files = [f for f in files if f.suffix == '.py']
        total_py_lines = 0
        total_functions = 0
        total_classes = 0
        for f in py_files[:max_files]:
            try:
                source = f.read_text(encoding="utf-8", errors="replace")
                total_py_lines += len(source.splitlines())
                try:
                    tree = ast.parse(source)
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            total_functions += 1
                        elif isinstance(node, ast.ClassDef):
                            total_classes += 1
                except SyntaxError:
                    pass
            except Exception:
                pass

        lines = [
            f"Project: {root.absolute()}",
            f"Total files: {len(files)}",
            f"Total dirs: {len(dirs)}",
            f"Python files: {len(py_files)}",
            f"Python lines: {total_py_lines:,}",
            f"Functions: {total_functions:,}",
            f"Classes: {total_classes:,}",
            "",
            "File types:",
        ]
        for ext, count in ext_count.most_common(20):
            lines.append(f"  {ext or '(no ext)'}: {count}")

        lines.append("")
        lines.append("Structure (top 3 levels):")
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

        lines = source.splitlines()
        ext = p.suffix.lower()

        stats = CodeAnalysis._compute_stats(source, lines, ext)
        security_issues = CodeAnalysis._scan_security(source, lines)
        code_smells = CodeAnalysis._scan_code_smells(source, lines)
        quality_report = CodeAnalysis._compute_quality(source, lines, ext)

        report = [
            f"## Code Review: {file_path}",
            "",
            f"### Statistics",
            f"- Total lines: {stats['lines']:,} | Code: {stats['code_lines']:,} | Comments: {stats['comment_lines']:,} | Blank: {stats['blank_lines']:,}",
            f"- Functions: {stats['functions']:,} | Classes: {stats['classes']:,}",
            f"- Imports: {stats['imports']:,} | Avg line length: {stats['avg_line_length']:.1f} chars",
            f"- Max line length: {stats['max_line_length']} chars",
            "",
        ]

        report.append(f"### Quality Score: {quality_report['score']}/100 ({quality_report['rating']})")
        report.append(f"  Cyclomatic Complexity: {quality_report['complexity']}")
        report.append(f"  Maintainability: {quality_report['maintainability']}")
        report.append(f"  Nesting Depth: {quality_report['nesting_depth']}")
        report.append("")

        if security_issues:
            report.append(f"### Security Issues ({len(security_issues)})")
            crit = [s for s in security_issues if s[2] == 'CRITICAL']
            high = [s for s in security_issues if s[2] == 'HIGH']
            med = [s for s in security_issues if s[2] == 'MEDIUM']
            if crit:
                report.append(f"  **CRITICAL ({len(crit)}):**")
                for line_no, desc, sev in crit:
                    report.append(f"    - Line {line_no}: {desc}")
            if high:
                report.append(f"  **HIGH ({len(high)}):**")
                for line_no, desc, sev in high:
                    report.append(f"    - Line {line_no}: {desc}")
            if med:
                report.append(f"  **MEDIUM ({len(med)}):**")
                for line_no, desc, sev in med:
                    report.append(f"    - Line {line_no}: {desc}")
            report.append("")

        if code_smells:
            report.append(f"### Code Smells ({len(code_smells)})")
            for line_no, desc, sev in code_smells[:15]:
                report.append(f"  - Line {line_no}: {desc}")
            if len(code_smells) > 15:
                report.append(f"  ... and {len(code_smells) - 15} more")
            report.append("")

        suggestions = CodeAnalysis._generate_suggestions(security_issues, code_smells, quality_report)
        if suggestions:
            report.append("### Suggestions")
            for s in suggestions:
                report.append(f"  - {s}")
            report.append("")

        return "\n".join(report)

    @staticmethod
    def _compute_stats(source: str, lines: list, ext: str) -> dict:
        stats = {
            "lines": len(lines),
            "code_lines": 0,
            "comment_lines": 0,
            "blank_lines": 0,
            "functions": 0,
            "classes": 0,
            "imports": 0,
            "max_line_length": 0,
            "avg_line_length": 0,
        }

        total_length = 0
        for line in lines:
            stripped = line.strip()
            line_len = len(line.rstrip('\n'))
            stats["max_line_length"] = max(stats["max_line_length"], line_len)
            total_length += line_len

            if not stripped:
                stats["blank_lines"] += 1
            elif stripped.startswith("#") or stripped.startswith("//") or stripped.startswith("/*"):
                stats["comment_lines"] += 1
            else:
                stats["code_lines"] += 1

            if stripped.startswith("import ") or stripped.startswith("from "):
                stats["imports"] += 1

        stats["avg_line_length"] = total_length / max(len(lines), 1)

        if ext == ".py":
            try:
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        stats["functions"] += 1
                    elif isinstance(node, ast.ClassDef):
                        stats["classes"] += 1
            except SyntaxError:
                pass

        return stats

    @staticmethod
    def _scan_security(source: str, lines: list) -> list[tuple]:
        issues = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            for pattern, desc, severity in CodeAnalysis.SECURITY_PATTERNS:
                if re.search(pattern, stripped):
                    issues.append((i, desc, severity))
                    break
        return issues

    @staticmethod
    def _scan_code_smells(source: str, lines: list) -> list[tuple]:
        issues = []
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped:
                continue
            for pattern, desc, severity in CodeAnalysis.CODE_SMELL_PATTERNS:
                if re.search(pattern, stripped):
                    issues.append((i, desc, severity))
                    break
        return issues

    @staticmethod
    def _compute_quality(source: str, lines: list, ext: str) -> dict:
        complexity = 0
        nesting_depth = 0
        func_count = 0
        class_count = 0
        comment_ratio = 0

        code_lines = 0
        comment_lines = 0
        blank_lines = 0

        for line in lines:
            stripped = line.strip()
            if not stripped:
                blank_lines += 1
            elif stripped.startswith('#'):
                comment_lines += 1
            else:
                code_lines += 1

        comment_ratio = (comment_lines / max(code_lines, 1)) * 100

        if ext == ".py":
            try:
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        func_count += 1
                        base = node.lineno
                        complexity += 1
                        for child in ast.walk(node):
                            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor,
                                                  ast.ExceptHandler, ast.With, ast.AsyncWith,
                                                  ast.Assert)):
                                complexity += 1
                    elif isinstance(node, ast.ClassDef):
                        class_count += 1
                        try:
                            for child in ast.walk(node):
                                if isinstance(child, (ast.If, ast.While, ast.For)):
                                    depth = CodeAnalysis._get_nesting_depth(child, node)
                                    nesting_depth = max(nesting_depth, depth)
                        except RecursionError:
                            pass
            except SyntaxError:
                pass

        for line in lines:
            depth = len(line) - len(line.lstrip())
            if depth > 0:
                nesting_depth = max(nesting_depth, depth // 4)

        maintainability = min(100, max(0,
            (comment_ratio * 0.3) +
            (1.0 - (complexity / max(func_count, 1)) * 0.05) * 50 +
            (1.0 - (nesting_depth / 10)) * 20 +
            (1.0 - len(lines) / 2000) * 10
        ))

        score_penalties = 0
        for line in lines:
            stripped = line.strip()
            if len(line) > 100:
                score_penalties += 2
            if 'TODO' in stripped or 'FIXME' in stripped:
                score_penalties += 5
            if re.search(r'except\s*:', stripped):
                score_penalties += 3
            if re.search(r'(?<!\w)print\(', stripped):
                score_penalties += 1

        score = max(0, min(100, maintainability - score_penalties))

        if score >= 90:
            rating = "EXCELLENT"
        elif score >= 75:
            rating = "GOOD"
        elif score >= 55:
            rating = "FAIR"
        elif score >= 35:
            rating = "POOR"
        else:
            rating = "BAD"

        return {
            "score": round(score, 1),
            "rating": rating,
            "complexity": complexity,
            "maintainability": round(maintainability, 1),
            "nesting_depth": nesting_depth,
            "func_comment_ratio": f"{comment_ratio:.1f}%",
        }

    @staticmethod
    def _get_nesting_depth(node, root, depth=0):
        parent = getattr(node, 'parent', None)
        if parent is None or node is root:
            return depth
        if isinstance(parent, (ast.If, ast.While, ast.For, ast.Try, ast.With,
                                ast.FunctionDef, ast.AsyncFunctionDef)):
            depth += 1
        return CodeAnalysis._get_nesting_depth(parent, root, depth)

    @staticmethod
    def _generate_suggestions(security: list, smells: list, quality: dict) -> list[str]:
        suggestions = []
        if security:
            crit_count = sum(1 for s in security if s[2] == 'CRITICAL')
            if crit_count > 0:
                suggestions.append(f"Fix {crit_count} critical security issues immediately")
            high_count = sum(1 for s in security if s[2] == 'HIGH')
            if high_count > 0:
                suggestions.append(f"Address {high_count} high severity security issues")

        if quality['score'] < 55:
            suggestions.append("Major refactoring needed - code quality is below threshold")
        if quality['complexity'] > 30:
            suggestions.append("High complexity - break down large functions and reduce nesting")
        if quality['nesting_depth'] > 5:
            suggestions.append("Deep nesting - extract inner logic into separate functions")

        bare_excepts = sum(1 for s in smells if 'bare except' in s[1].lower())
        if bare_excepts > 0:
            suggestions.append(f"Replace {bare_excepts} bare except clauses with specific exception types")

        if not suggestions:
            suggestions.append("Code looks good! Continue following best practices.")

        return suggestions

    @staticmethod
    def analyze_imports(file_path: str) -> str:
        p = Path(file_path)
        if not p.exists():
            return f"Error: File not found: {file_path}"

        try:
            source = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"Error reading file: {e}"

        stdlib = {"os", "sys", "json", "re", "math", "datetime", "pathlib", "collections",
                   "typing", "functools", "itertools", "hashlib", "subprocess", "time",
                   "random", "ast", "argparse", "copy", "enum", "io", "textwrap", "uuid",
                   "asyncio", "threading", "multiprocessing", "socket", "http", "urllib",
                   "xml", "html", "csv", "sqlite3", "logging", "unittest", "dataclasses",
                   "base64", "binascii", "calendar", "cmath", "contextlib", "decimal",
                   "difflib", "dis", "fileinput", "fractions", "getopt", "getpass",
                   "glob", "gzip", "inspect", "locale", "lzma", "mmap", "numbers",
                   "operator", "os.path", "pickle", "pkgutil", "platform", "pprint",
                   "profile", "pstats", "queue", "reprlib", "shlex", "shutil",
                   "signal", "smtplib", "sndhdr", "spwd", "sqlite3", "ssl",
                   "stat", "statistics", "string", "struct", "tarfile", "tempfile",
                   "test", "textwrap", "threading", "timeit", "tkinter", "tokenize",
                   "trace", "traceback", "tracemalloc", "tty", "turtle", "types",
                   "unicodedata", "venv", "warnings", "wave", "weakref", "webbrowser",
                   "winreg", "winsound", "wsgiref", "xmlrpc", "zipapp", "zipfile",
                   "zipimport", "zlib", "importlib", "pkgutil", "pdb", "doctest",
                   "__future__", "builtins", "abc", "array", "atexit", "bisect"}

        imports = {"stdlib": [], "third_party": [], "local": [], "unused": []}

        try:
            tree = ast.parse(source)
            used_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    used_names.add(node.id)
                elif isinstance(node, ast.Attribute):
                    used_names.add(node.attr)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.name.split(".")[0]
                        cat = "stdlib" if name in stdlib else "third_party" if name != "config" else "local"
                        if alias.name in imports[cat]:
                            continue
                        imports[cat].append(alias.name)

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        name = node.module.split(".")[0]
                        cat = "stdlib" if name in stdlib else "third_party"
                        imported_items = [a.name for a in node.names]
                        for item in imported_items:
                            full = f"{node.module}.{item}"
                            imports.setdefault(cat, []).append(full)

        except SyntaxError:
            for line in source.splitlines():
                stripped = line.strip()
                m = re.match(r'^import\s+(\S+)', stripped)
                if m:
                    name = m.group(1).split(".")[0]
                    cat = "stdlib" if name in stdlib else "third_party" if name != "config" else "local"
                    imports[cat].append(m.group(1))
                m = re.match(r'^from\s+(\S+)\s+import', stripped)
                if m:
                    name = m.group(1).split(".")[0]
                    cat = "stdlib" if name in stdlib else "third_party" if name != "config" else "local"
                    imports[cat].append(name)

        total = sum(len(v) for v in imports.values())
        lines_out = [
            f"## Import Analysis: {file_path}",
            f"Total: {total} imports",
            "",
        ]

        for cat, label in [("stdlib", "Standard Library"), ("third_party", "Third Party"), ("local", "Local")]:
            if imports[cat]:
                lines_out.append(f"### {label} ({len(imports[cat])})")
                for name in sorted(set(imports[cat])):
                    lines_out.append(f"  - `{name}`")
                lines_out.append("")

        return "\n".join(lines_out)

    @staticmethod
    def code_refactor(file_path: str, instructions: str = "") -> str:
        p = Path(file_path)
        if not p.exists():
            return f"Error: File not found: {file_path}"

        try:
            source = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"Error reading file: {e}"

        lines = source.splitlines()
        improvements = []

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped:
                continue

            if len(line) > 100:
                improvements.append(f"Line {i}: {len(line)} chars (max 100) - break into multiple lines")

            if re.search(r'(?<!\w)print\(', stripped) and not re.search(r'logger|logging', source):
                improvements.append(f"Line {i}: Use logging instead of print()")

            if re.search(r'except\s*:', stripped):
                improvements.append(f"Line {i}: Replace bare except with specific exception types")

            if 'import *' in stripped:
                improvements.append(f"Line {i}: Replace wildcard import with explicit imports")

            if re.search(r'(?<!\w)eval\(', stripped):
                improvements.append(f"Line {i}: eval() is a security risk - refactor with safe alternatives")

            if re.search(r'(?<!\w)exec\(', stripped):
                improvements.append(f"Line {i}: exec() is a security risk - refactor")

            if re.search(r'(?<!\w)input\(', stripped):
                improvements.append(f"Line {i}: input() usage - ensure proper validation")

            if re.search(r'#\s*(TODO|FIXME|HACK|XXX)', stripped):
                improvements.append(f"Line {i}: Incomplete code marker found")

        if not instructions:
            instructions = "Improve code quality, readability, and maintainability"

        report = [
            f"## Refactoring Analysis: {file_path}",
            f"Instructions: {instructions}",
            "",
            f"Found {len(improvements)} potential improvements:",
        ]
        for imp in improvements[:20]:
            report.append(f"  - {imp}")
        if len(improvements) > 20:
            report.append(f"  ... and {len(improvements) - 20} more")
        if not improvements:
            report.append("  Code looks good! No obvious refactoring needed.")

        return "\n".join(report)

    @staticmethod
    def complexity_metrics(file_path: str) -> str:
        p = Path(file_path)
        if not p.exists():
            return f"Error: File not found: {file_path}"

        try:
            source = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"Error reading file: {e}"

        lines = source.splitlines()
        ext = p.suffix.lower()

        stats = CodeAnalysis._compute_stats(source, lines, ext)
        quality = CodeAnalysis._compute_quality(source, lines, ext)

        report = [
            f"## Complexity Metrics: {file_path}",
            "",
            "### Raw Metrics",
            f"  Lines: {stats['lines']:,} (Code: {stats['code_lines']:,}, Comments: {stats['comment_lines']:,}, Blank: {stats['blank_lines']:,})",
            f"  Functions: {stats['functions']:,} | Classes: {stats['classes']:,} | Imports: {stats['imports']:,}",
            f"  Max line: {stats['max_line_length']} chars | Avg line: {stats['avg_line_length']:.1f} chars",
            "",
            "### Quality Assessment",
            f"  Score: {quality['score']}/100 ({quality['rating']})",
            f"  Cyclomatic Complexity: {quality['complexity']}",
            f"  Maintainability Index: {quality['maintainability']}",
            f"  Max Nesting Depth: {quality['nesting_depth']}",
            f"  Comment Ratio: {quality['func_comment_ratio']}",
            "",
        ]

        if ext == ".py":
            try:
                tree = ast.parse(source)
                func_details = []
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        func_complexity = 1
                        for child in ast.walk(node):
                            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler,
                                                  ast.With, ast.Assert)):
                                func_complexity += 1
                        func_lines = (node.end_lineno - node.lineno + 1) if hasattr(node, 'end_lineno') else 0
                        func_details.append((node.name, func_complexity, func_lines, node.lineno))

                if func_details:
                    report.append("### Function Complexity Breakdown")
                    report.append(f"  {'Function':<25} {'Complexity':<12} {'Lines':<8} {'Line':<6}")
                    report.append(f"  {'─'*25} {'─'*12} {'─'*8} {'─'*6}")
                    for name, comp, f_lines, line_no in sorted(func_details, key=lambda x: x[1], reverse=True):
                        report.append(f"  {name:<25} {comp:<12} {f_lines:<8} {line_no:<6}")
                    avg_comp = sum(f[1] for f in func_details) / max(len(func_details), 1)
                    report.append(f"\n  Average function complexity: {avg_comp:.1f}")
                    high_comp = [f for f in func_details if f[1] > 10]
                    if high_comp:
                        report.append(f"  Functions with high complexity (>10): {len(high_comp)}")
                        for name, comp, fl, ln in high_comp:
                            report.append(f"    - {name} (complexity: {comp}, line {ln})")
            except SyntaxError:
                report.append("  (Could not parse AST for function details)")

        return "\n".join(report)

    @staticmethod
    def analyze_security(file_path: str) -> str:
        p = Path(file_path)
        if not p.exists():
            return f"Error: File not found: {file_path}"

        try:
            source = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"Error reading file: {e}"

        lines = source.splitlines()
        issues = CodeAnalysis._scan_security(source, lines)

        report = [
            f"## Security Scan: {file_path}",
            "",
        ]

        if not issues:
            report.append("  ✅ No security issues detected!")
            return "\n".join(report)

        by_severity = {"CRITICAL": [], "HIGH": [], "MEDIUM": [], "LOW": [], "INFO": []}
        for line_no, desc, sev in issues:
            by_severity.setdefault(sev, []).append((line_no, desc))

        for sev, label in [("CRITICAL", "Critical"), ("HIGH", "High"), ("MEDIUM", "Medium"), ("LOW", "Low")]:
            if by_severity.get(sev):
                report.append(f"  ### {label} ({len(by_severity[sev])})")
                for line_no, desc in by_severity[sev]:
                    report.append(f"    - Line {line_no}: {desc}")
                report.append("")

        total = len(issues)
        weighted = (
            len(by_severity.get("CRITICAL", [])) * 10 +
            len(by_severity.get("HIGH", [])) * 5 +
            len(by_severity.get("MEDIUM", [])) * 2
        )
        if weighted >= 20:
            risk = "CRITICAL"
        elif weighted >= 10:
            risk = "HIGH"
        elif weighted >= 5:
            risk = "MEDIUM"
        elif weighted > 0:
            risk = "LOW"
        else:
            risk = "NONE"

        report.append(f"  **Overall Risk: {risk}** (score: {weighted})")
        return "\n".join(report)

    @staticmethod
    def dependency_graph(path: str = "") -> str:
        root = Path(path) if path else Path.cwd()
        if not root.exists():
            return f"Error: Path not found: {path}"

        dependencies = {}
        local_modules = set()

        for py_file in root.rglob("*.py"):
            if any(x in str(py_file) for x in ("__pycache__", ".venv", ".git", "venv")):
                continue
            local_modules.add(py_file.stem)
            if py_file.stem not in dependencies:
                dependencies[py_file.stem] = []

        for py_file in root.rglob("*.py"):
            if any(x in str(py_file) for x in ("__pycache__", ".venv", ".git", "venv")):
                continue
            module_name = py_file.stem
            try:
                source = py_file.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            name = alias.name.split(".")[0]
                            if name in local_modules and name != module_name:
                                if name not in dependencies[module_name]:
                                    dependencies[module_name].append(name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            name = node.module.split(".")[0]
                            if name in local_modules and name != module_name:
                                if name not in dependencies[module_name]:
                                    dependencies[module_name].append(name)
            except Exception:
                continue

        deps_with_deps = {k: v for k, v in dependencies.items() if v}
        orphaned = [k for k in local_modules if k not in deps_with_deps and not any(k in v for v in deps_with_deps.values())]
        bad_deps = [k for k, v in deps_with_deps.items() if k in v]

        lines = [
            f"## Dependency Graph: {root.name}",
            f"Total modules: {len(local_modules)}",
            f"Modules with dependencies: {len(deps_with_deps)}",
            f"Orphaned modules: {len(orphaned)}",
            "",
        ]

        if deps_with_deps:
            lines.append("### Dependencies")
            for module, deps in sorted(deps_with_deps.items()):
                lines.append(f"  {module} → {', '.join(sorted(deps))}")
            lines.append("")

        if bad_deps:
            lines.append("### ⚠️ Self-Dependencies")
            for m in bad_deps:
                lines.append(f"  {m} depends on itself!")

        if orphaned:
            lines.append("\n### Orphaned Modules (no dependents)")
            for m in sorted(orphaned):
                lines.append(f"  {m}")

        return "\n".join(lines)

    @staticmethod
    def analyze_code_quality(file_path: str) -> str:
        p = Path(file_path)
        if not p.exists():
            return f"Error: File not found: {file_path}"

        try:
            source = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"Error reading file: {e}"

        lines = source.splitlines()
        ext = p.suffix.lower()

        if ext != ".py":
            return f"Quality analysis only supports Python files"

        quality = CodeAnalysis._compute_quality(source, lines, ext)
        stats = CodeAnalysis._compute_stats(source, lines, ext)
        security = CodeAnalysis._scan_security(source, lines)
        smells = CodeAnalysis._scan_code_smells(source, lines)
        suggestions = CodeAnalysis._generate_suggestions(security, smells, quality)

        report = [
            f"## Code Quality Report: {file_path}",
            "",
            f"### Overall Quality: {quality['score']}/100 ({quality['rating']})",
            "",
            "### Metrics Summary",
            f"  Lines: {stats['lines']} (Code: {stats['code_lines']}, Comments: {stats['comment_lines']}, Blank: {stats['blank_lines']})",
            f"  Functions: {stats['functions']} | Classes: {stats['classes']}",
            f"  Cyclomatic Complexity: {quality['complexity']}",
            f"  Maintainability Index: {quality['maintainability']}",
            f"  Max Nesting Depth: {quality['nesting_depth']}",
            f"  Comment Ratio: {quality['func_comment_ratio']}",
            "",
            f"### Issues Found",
            f"  Security: {len(security)}",
            f"  Code Smells: {len(smells)}",
            "",
            "### Recommendations",
        ]
        for s in suggestions:
            report.append(f"  - {s}")

        return "\n".join(report)

    @staticmethod
    def generate_test(file_path: str, function_name: str = "") -> str:
        p = Path(file_path)
        if not p.exists():
            return f"Error: File not found: {file_path}"

        try:
            source = p.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            return f"Error reading file: {e}"

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return f"Error: Syntax error in file: {e}"

        functions = []
        classes = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append(node)
            elif isinstance(node, ast.ClassDef):
                classes.append(node)

        if function_name:
            functions = [f for f in functions if f.name == function_name]
            if not functions:
                return f"Function '{function_name}' not found in {file_path}"

        test_cases = []
        for func in functions:
            args = [a.arg for a in func.args.args if a.arg != 'self']
            has_self = any(a.arg == 'self' for a in func.args.args)
            is_async = isinstance(func, ast.AsyncFunctionDef)

            test_name = f"test_{func.name}"

            if has_self:
                test_cases.append(f"""
    def {test_name}(self):
        \"\"\"Test {func.name} - basic functionality\"\"\"
        pass""")
            else:
                params = ', '.join(args) if args else ''
                test_cases.append(f"""
    def {test_name}():
        \"\"\"Test {func.name} - basic functionality\"\"\"
        pass""")

        module_name = p.stem

        report = [
            f"import pytest",
            f"from {module_name} import {', '.join(f.name for f in functions[:5])}",
            "",
            "",
            f"class Test{module_name.title()}:",
            "",
            f"    \"\"\"Tests for {module_name}\"\"\"",
            "",
        ]

        for test in test_cases:
            report.append(test)
            report.append("")

        if test_cases:
            return "\n".join(report)
        else:
            return f"No testable functions found in {file_path}"
