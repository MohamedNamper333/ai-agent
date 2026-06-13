"""tools/deep_analyzer.py — Deep Strategic Analyzer

Multi-pass analysis engine that finds what single-pass tools miss.

The Agent Fable 5 approach (21 vulnerabilities in complex systems):
  Pass 1: Surface scan   — obvious issues (fast, AST-based)
  Pass 2: Semantic scan  — logic flaws (LLM-based)
  Pass 3: Attack surface — security threats (pattern matching)
  Pass 4: Architecture   — structural weaknesses (graph analysis)
  Pass 5: Cross-cutting  — issues that span multiple files

Each pass builds on findings from the previous one.
"""
from __future__ import annotations

import ast
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────
#  Findings
# ─────────────────────────────────────────────
@dataclass
class Finding:
    finding_id: str
    category: str       # security | logic | performance | architecture | style
    severity: str       # critical | high | medium | low | info
    title: str
    description: str
    location: str       # file:line or function name
    evidence: str       # The actual code/text that triggered this
    recommendation: str
    cwe_id: str = ""    # CWE ID for security findings
    pass_number: int = 1

    def to_text(self) -> str:
        sev_icons = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵", "info": "⚪"}
        icon = sev_icons.get(self.severity, "•")
        lines = [
            f"{icon} [{self.severity.upper()}] {self.title}",
            f"   Location: {self.location}",
            f"   {self.description}",
        ]
        if self.evidence:
            lines.append(f"   Evidence: `{self.evidence[:80]}`")
        lines.append(f"   Fix: {self.recommendation}")
        if self.cwe_id:
            lines.append(f"   CWE: {self.cwe_id}")
        return "\n".join(lines)


@dataclass
class AnalysisResult:
    target: str          # file path or code identifier
    findings: list[Finding]
    passes_run: int
    total_lines: int
    analysis_time_ms: float
    risk_score: float    # 0.0 (safe) - 10.0 (critical)
    summary: str

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "critical")

    @property
    def high_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "high")

    def to_report(self) -> str:
        sev_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
        sorted_findings = sorted(self.findings, key=lambda f: sev_order.get(f.severity, 5))

        lines = [
            "═" * 65,
            f"DEEP ANALYSIS REPORT — {self.target}",
            "═" * 65,
            f"Risk Score: {self.risk_score:.1f}/10 | "
            f"Findings: {len(self.findings)} "
            f"(Critical: {self.critical_count}, High: {self.high_count})",
            f"Lines analyzed: {self.total_lines} | "
            f"Passes: {self.passes_run} | "
            f"Time: {self.analysis_time_ms:.0f}ms",
            "─" * 65,
            self.summary,
            "─" * 65,
            f"FINDINGS ({len(self.findings)}):",
            "",
        ]
        for i, f in enumerate(sorted_findings, 1):
            lines.append(f"{i}. {f.to_text()}")
            lines.append("")
        return "\n".join(lines)


# ─────────────────────────────────────────────
#  Deep Analyzer
# ─────────────────────────────────────────────
class DeepAnalyzer:
    """
    Multi-pass analysis engine.
    Runs up to 5 passes, each finding different classes of issues.
    """

    def __init__(self, model=None):
        self._model = model
        self._finding_counter = 0

    def _next_id(self) -> str:
        self._finding_counter += 1
        return f"F{self._finding_counter:03d}"

    # ─────────────────────────────────────────
    #  Public interface
    # ─────────────────────────────────────────
    def analyze_code(self, code: str, filename: str = "<code>") -> AnalysisResult:
        """Full multi-pass analysis of a code string."""
        start = time.time()
        lines = code.split("\n")
        findings: list[Finding] = []

        # Pass 1: Surface scan (AST + syntax)
        findings.extend(self._pass1_surface(code, filename))

        # Pass 2: Security patterns
        findings.extend(self._pass2_security(code, filename))

        # Pass 3: Logic flaws (pattern-based)
        findings.extend(self._pass3_logic(code, filename))

        # Pass 4: Architecture / structure
        findings.extend(self._pass4_architecture(code, filename))

        # Pass 5: LLM semantic analysis (if model available)
        if self._model:
            findings.extend(self._pass5_llm_semantic(code, filename, findings))

        # Deduplicate
        findings = self._deduplicate(findings)

        risk = self._compute_risk(findings)
        summary = self._generate_summary(findings, risk)
        elapsed = (time.time() - start) * 1000

        return AnalysisResult(
            target=filename,
            findings=findings,
            passes_run=5 if self._model else 4,
            total_lines=len(lines),
            analysis_time_ms=elapsed,
            risk_score=risk,
            summary=summary,
        )

    def analyze_file(self, path: str) -> AnalysisResult:
        """Analyze a file from disk."""
        p = Path(path)
        if not p.exists():
            return AnalysisResult(
                target=path, findings=[], passes_run=0,
                total_lines=0, analysis_time_ms=0, risk_score=0,
                summary=f"File not found: {path}",
            )
        try:
            code = p.read_text(encoding="utf-8", errors="replace")
            return self.analyze_code(code, p.name)
        except Exception as e:
            return AnalysisResult(
                target=path, findings=[], passes_run=0,
                total_lines=0, analysis_time_ms=0, risk_score=0,
                summary=f"Read error: {e}",
            )

    def analyze_project(self, project_path: str) -> list[AnalysisResult]:
        """Analyze all Python files in a project directory."""
        base = Path(project_path)
        results = []
        for py_file in sorted(base.rglob("*.py")):
            if any(part.startswith(".") or part == "__pycache__" for part in py_file.parts):
                continue
            results.append(self.analyze_file(str(py_file)))
        return results

    # ─────────────────────────────────────────
    #  Pass 1: Surface Scan
    # ─────────────────────────────────────────
    def _pass1_surface(self, code: str, filename: str) -> list[Finding]:
        findings = []
        lines = code.split("\n")

        # Syntax check
        try:
            ast.parse(code)
        except SyntaxError as e:
            findings.append(Finding(
                finding_id=self._next_id(), category="style",
                severity="critical", pass_number=1,
                title="Syntax Error",
                description=f"File cannot be parsed: {e}",
                location=f"{filename}:{e.lineno}",
                evidence=str(e.text or "")[:60],
                recommendation="Fix syntax error before any other analysis",
            ))
            return findings

        # Bare except
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped == "except:" or stripped == "except :":
                findings.append(Finding(
                    finding_id=self._next_id(), category="logic",
                    severity="high", pass_number=1,
                    title="Bare except clause",
                    description="Catches ALL exceptions including KeyboardInterrupt and SystemExit",
                    location=f"{filename}:{i}",
                    evidence=stripped,
                    recommendation="Use `except Exception as e:` or specific exception types",
                ))

        # TODO/FIXME/HACK comments
        todo_pattern = re.compile(r'#\s*(TODO|FIXME|HACK|XXX|BUG|SECURITY)\b', re.IGNORECASE)
        for i, line in enumerate(lines, 1):
            m = todo_pattern.search(line)
            if m:
                tag = m.group(1).upper()
                severity = "high" if tag in ("SECURITY", "BUG", "FIXME") else "low"
                findings.append(Finding(
                    finding_id=self._next_id(), category="style",
                    severity=severity, pass_number=1,
                    title=f"Unresolved {tag} comment",
                    description="Known issue or incomplete implementation",
                    location=f"{filename}:{i}",
                    evidence=line.strip()[:70],
                    recommendation=f"Resolve or ticket the {tag} before production",
                ))

        # Mutable default arguments
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for default in node.args.defaults:
                        if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                            findings.append(Finding(
                                finding_id=self._next_id(), category="logic",
                                severity="high", pass_number=1,
                                title="Mutable default argument",
                                description=(
                                    f"Function `{node.name}` has a mutable default "
                                    f"(list/dict/set). Shared across all calls — common bug."
                                ),
                                location=f"{filename}:{node.lineno}",
                                evidence=f"def {node.name}(..., default=[])",
                                recommendation="Use `None` as default and assign inside function",
                            ))
        except SyntaxError:
            pass

        return findings

    # ─────────────────────────────────────────
    #  Pass 2: Security Patterns
    # ─────────────────────────────────────────

    SECURITY_PATTERNS = [
        (r'\beval\s*\(', "critical", "Use of eval()", "CWE-95",
         "eval() executes arbitrary code. Use ast.literal_eval() for data, or redesign."),
        (r'\bexec\s*\(', "critical", "Use of exec()", "CWE-95",
         "exec() executes arbitrary code. Replace with a safer alternative."),
        (r'pickle\.loads?\s*\(', "critical", "Pickle deserialization", "CWE-502",
         "Pickle can execute arbitrary code during deserialization. Use JSON or MessagePack."),
        (r'subprocess\.call\s*\(.+shell\s*=\s*True', "critical", "Shell injection risk", "CWE-78",
         "shell=True with user input allows command injection. Use list form instead."),
        (r'os\.system\s*\(', "high", "os.system() usage", "CWE-78",
         "Use subprocess.run() with shell=False for better control and security."),
        (r'hashlib\.md5\s*\(', "high", "Weak hash (MD5)", "CWE-327",
         "MD5 is cryptographically broken. Use SHA-256 or SHA-3 for security."),
        (r'hashlib\.sha1\s*\(', "medium", "Weak hash (SHA-1)", "CWE-327",
         "SHA-1 is weak. Use SHA-256 or bcrypt for passwords."),
        (r'random\.\w+\s*\(', "medium", "Non-cryptographic random", "CWE-338",
         "Use secrets module for security-sensitive randomness."),
        (r'sql\s*=\s*["\'].*%s', "critical", "SQL injection risk", "CWE-89",
         "String formatting in SQL queries allows injection. Use parameterized queries."),
        (r'sql\s*=\s*f["\'].*\{', "critical", "SQL injection risk (f-string)", "CWE-89",
         "f-string SQL queries allow injection. Use parameterized queries."),
        (r'password\s*=\s*["\'][^"\']+["\']', "critical", "Hardcoded password", "CWE-259",
         "Never hardcode passwords. Use environment variables or a secrets manager."),
        (r'api_key\s*=\s*["\'][^"\']{10,}["\']', "critical", "Hardcoded API key", "CWE-312",
         "Never hardcode API keys. Use environment variables."),
        (r'secret\s*=\s*["\'][^"\']{8,}["\']', "high", "Hardcoded secret", "CWE-312",
         "Use environment variables for secrets."),
        (r'verify\s*=\s*False', "high", "SSL verification disabled", "CWE-295",
         "Disabling SSL verification allows MITM attacks. Remove verify=False."),
        (r'DEBUG\s*=\s*True', "medium", "Debug mode enabled", "CWE-215",
         "Ensure DEBUG=False in production. Debug mode exposes stack traces."),
        (r'open\s*\(.+["\']w["\']', "low", "File write without error handling", "",
         "Wrap file writes in try/except and use context managers."),
    ]

    def _pass2_security(self, code: str, filename: str) -> list[Finding]:
        findings = []
        lines = code.split("\n")

        for pattern, severity, title, cwe, recommendation in self.SECURITY_PATTERNS:
            for i, line in enumerate(lines, 1):
                if re.search(pattern, line, re.IGNORECASE):
                    findings.append(Finding(
                        finding_id=self._next_id(), category="security",
                        severity=severity, pass_number=2,
                        title=title,
                        description=f"Potentially dangerous pattern detected",
                        location=f"{filename}:{i}",
                        evidence=line.strip()[:80],
                        recommendation=recommendation,
                        cwe_id=cwe,
                    ))
        return findings

    # ─────────────────────────────────────────
    #  Pass 3: Logic Flaws
    # ─────────────────────────────────────────
    def _pass3_logic(self, code: str, filename: str) -> list[Finding]:
        findings = []
        lines = code.split("\n")

        # Division without zero check
        div_pattern = re.compile(r'[^/]/(?!/)[^/=]')
        for i, line in enumerate(lines, 1):
            if div_pattern.search(line) and "ZeroDivision" not in code[max(0, i-5*80):i*80]:
                findings.append(Finding(
                    finding_id=self._next_id(), category="logic",
                    severity="medium", pass_number=3,
                    title="Potential ZeroDivisionError",
                    description="Division without zero-check",
                    location=f"{filename}:{i}",
                    evidence=line.strip()[:60],
                    recommendation="Add `if denominator != 0:` or use try/except ZeroDivisionError",
                ))
                break  # Only flag once per file for this

        # Comparing to None with ==
        none_eq = re.compile(r'\bif\s+\w+\s*==\s*None\b|\bif\s+None\s*==\s*\w+\b')
        for i, line in enumerate(lines, 1):
            if none_eq.search(line):
                findings.append(Finding(
                    finding_id=self._next_id(), category="style",
                    severity="low", pass_number=3,
                    title="Use `is None` instead of `== None`",
                    description="== None works but `is None` is the Pythonic way and avoids __eq__ issues",
                    location=f"{filename}:{i}",
                    evidence=line.strip()[:60],
                    recommendation="Change `== None` to `is None`",
                ))

        # Empty except with pass
        empty_except = re.compile(r'except.*:\s*\n\s*pass\s*$', re.MULTILINE)
        for m in empty_except.finditer(code):
            lineno = code[:m.start()].count("\n") + 1
            findings.append(Finding(
                finding_id=self._next_id(), category="logic",
                severity="high", pass_number=3,
                title="Silent exception suppression",
                description="Exception caught and silently ignored with `pass`",
                location=f"{filename}:{lineno}",
                evidence="except ...: pass",
                recommendation="Log the exception or handle it properly. Silent failures are hard to debug.",
            ))

        # String concatenation in loop
        concat_loop = re.compile(r'for\s+\w+\s+in.*:\s*\n(?:.*\n)*?\s*\w+\s*\+=\s*["\']')
        for m in concat_loop.finditer(code):
            lineno = code[:m.start()].count("\n") + 1
            findings.append(Finding(
                finding_id=self._next_id(), category="performance",
                severity="medium", pass_number=3,
                title="String concatenation in loop",
                description="O(n²) string building — creates new string object each iteration",
                location=f"{filename}:{lineno}",
                evidence="for ...: result += '...'",
                recommendation="Use list.append() + ''.join() or io.StringIO for O(n) performance",
            ))

        # Nested loops over same data
        nested_for = re.compile(r'for\s+(\w+)\s+in\s+(\w+).*:\s*\n.*for\s+\w+\s+in\s+\2')
        for m in nested_for.finditer(code):
            lineno = code[:m.start()].count("\n") + 1
            findings.append(Finding(
                finding_id=self._next_id(), category="performance",
                severity="high", pass_number=3,
                title="O(n²) nested loop over same collection",
                description=f"Nested iteration over `{m.group(2)}` — quadratic complexity",
                location=f"{filename}:{lineno}",
                evidence=m.group(0)[:60],
                recommendation="Use a dict/set for O(1) lookup, or rethink the algorithm",
            ))

        return findings

    # ─────────────────────────────────────────
    #  Pass 4: Architecture
    # ─────────────────────────────────────────
    def _pass4_architecture(self, code: str, filename: str) -> list[Finding]:
        findings = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return findings

        # God class (too many methods)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n for n in ast.walk(node) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                if len(methods) > 20:
                    findings.append(Finding(
                        finding_id=self._next_id(), category="architecture",
                        severity="high", pass_number=4,
                        title=f"God class: `{node.name}`",
                        description=f"Class has {len(methods)} methods — too many responsibilities",
                        location=f"{filename}:{node.lineno}",
                        evidence=f"class {node.name}: # {len(methods)} methods",
                        recommendation="Split into smaller, focused classes following Single Responsibility Principle",
                    ))

        # Large files
        lines = code.split("\n")
        if len(lines) > 800:
            findings.append(Finding(
                finding_id=self._next_id(), category="architecture",
                severity="high", pass_number=4,
                title="File too large",
                description=f"{len(lines)} lines in one file — hard to maintain",
                location=filename,
                evidence=f"{len(lines)} lines",
                recommendation="Split into multiple focused modules",
            ))

        # Circular import indicators
        import_names = set()
        file_stem = Path(filename).stem
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                import_names.add(node.module)
        # If a file imports itself (via relative import patterns)
        if any(file_stem in m for m in import_names):
            findings.append(Finding(
                finding_id=self._next_id(), category="architecture",
                severity="high", pass_number=4,
                title="Possible circular import",
                description=f"Module may import from itself or create circular dependency",
                location=filename,
                evidence=f"imports: {list(import_names)[:3]}",
                recommendation="Use dependency injection or restructure module boundaries",
            ))

        return findings

    # ─────────────────────────────────────────
    #  Pass 5: LLM Semantic Analysis
    # ─────────────────────────────────────────
    def _pass5_llm_semantic(
        self, code: str, filename: str, existing_findings: list[Finding]
    ) -> list[Finding]:
        if not self._model:
            return []

        known = "\n".join(f"- {f.title}" for f in existing_findings[:10])
        prompt = f"""You are a senior security and code quality engineer.

Analyze this Python code for issues the static analyzer MISSED.
Already found by static analysis:
{known}

CODE:
```python
{code[:3000]}
```

Find 3-7 ADDITIONAL issues focusing on:
- Business logic flaws
- Race conditions or concurrency issues
- Missing input validation
- Incorrect algorithm/data structure choice
- Hidden performance bottlenecks
- Missing edge cases

For each issue respond in this EXACT format (one per line group):
TITLE: [short title]
SEVERITY: critical|high|medium|low
CATEGORY: security|logic|performance|architecture
LOCATION: [function name or line indicator]
DESCRIPTION: [one sentence]
FIX: [one sentence recommendation]
---
"""
        try:
            response = self._model.generate(prompt, max_tokens=800)
            return self._parse_llm_findings(response, filename)
        except Exception:
            return []

    def _parse_llm_findings(self, text: str, filename: str) -> list[Finding]:
        findings = []
        blocks = text.split("---")
        valid_severities = {"critical", "high", "medium", "low", "info"}

        for block in blocks:
            if "TITLE:" not in block:
                continue
            try:
                data: dict[str, str] = {}
                for line in block.strip().split("\n"):
                    if ":" in line:
                        key, _, val = line.partition(":")
                        data[key.strip().upper()] = val.strip()

                severity = data.get("SEVERITY", "medium").lower()
                if severity not in valid_severities:
                    severity = "medium"

                if data.get("TITLE"):
                    findings.append(Finding(
                        finding_id=self._next_id(),
                        category=data.get("CATEGORY", "logic").lower(),
                        severity=severity,
                        pass_number=5,
                        title=data.get("TITLE", "Unknown issue"),
                        description=data.get("DESCRIPTION", ""),
                        location=f"{filename}: {data.get('LOCATION', 'unknown')}",
                        evidence="",
                        recommendation=data.get("FIX", "Review this code"),
                    ))
            except Exception:
                continue
        return findings

    # ─────────────────────────────────────────
    #  Helpers
    # ─────────────────────────────────────────
    def _deduplicate(self, findings: list[Finding]) -> list[Finding]:
        seen: set[str] = set()
        unique = []
        for f in findings:
            key = f"{f.title}:{f.location}"
            if key not in seen:
                seen.add(key)
                unique.append(f)
        return unique

    def _compute_risk(self, findings: list[Finding]) -> float:
        weights = {"critical": 3.0, "high": 1.5, "medium": 0.5, "low": 0.1, "info": 0.0}
        score = sum(weights.get(f.severity, 0) for f in findings)
        return min(10.0, score)

    def _generate_summary(self, findings: list[Finding], risk: float) -> str:
        if not findings:
            return "✅ No issues found. Code appears clean."

        by_sev: dict[str, int] = {}
        by_cat: dict[str, int] = {}
        for f in findings:
            by_sev[f.severity] = by_sev.get(f.severity, 0) + 1
            by_cat[f.category] = by_cat.get(f.category, 0) + 1

        risk_label = (
            "CRITICAL — immediate action required"
            if risk >= 7 else
            "HIGH — address before production"
            if risk >= 4 else
            "MEDIUM — schedule for next sprint"
            if risk >= 2 else
            "LOW — minor improvements"
        )

        top_cat = max(by_cat, key=by_cat.get) if by_cat else "general"
        return (
            f"Risk level: {risk_label} (score: {risk:.1f}/10)\n"
            f"Findings by severity: {by_sev}\n"
            f"Primary concern: {top_cat} ({by_cat.get(top_cat, 0)} issues)\n"
            f"Total: {len(findings)} findings across {len(by_cat)} categories"
        )
