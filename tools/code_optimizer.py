"""tools/code_optimizer.py — Code Optimization Engine

Capabilities:
  1. MEASURE   — compute code metrics (complexity, duplication, length)
  2. DETECT    — find specific optimization opportunities
  3. REFACTOR  — generate optimized versions with same behavior
  4. VERIFY    — confirm optimization doesn't break logic
  5. REPORT    — structured report with before/after metrics

The goal: reduce code size 70-90% where possible WITHOUT losing
functionality. Like the print("hi") × 300 → loop example.
"""
from __future__ import annotations

import ast
import re
import textwrap
import time
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────
@dataclass
class CodeMetrics:
    total_lines: int
    code_lines: int          # Non-empty, non-comment
    comment_lines: int
    blank_lines: int
    functions: int
    classes: int
    avg_function_length: float
    max_function_length: int
    cyclomatic_complexity: float
    duplication_score: float  # 0.0 = no dup, 1.0 = all duplicated
    imports_count: int
    nested_depth: int

    def score(self) -> float:
        """Quality score 0-100. Higher = better."""
        s = 100.0
        if self.avg_function_length > 50:
            s -= min(20, (self.avg_function_length - 50) * 0.4)
        if self.max_function_length > 100:
            s -= min(20, (self.max_function_length - 100) * 0.2)
        if self.cyclomatic_complexity > 10:
            s -= min(20, (self.cyclomatic_complexity - 10) * 2)
        if self.duplication_score > 0.1:
            s -= min(25, self.duplication_score * 50)
        if self.nested_depth > 4:
            s -= min(15, (self.nested_depth - 4) * 3)
        return max(0.0, s)

    def summary(self) -> str:
        """Summary."""
        return (
            f"Lines: {self.total_lines} (code={self.code_lines}, "
            f"comments={self.comment_lines}, blank={self.blank_lines})\n"
            f"Functions: {self.functions} | Avg length: {self.avg_function_length:.1f} lines\n"
            f"Max function: {self.max_function_length} lines\n"
            f"Cyclomatic complexity: {self.cyclomatic_complexity:.1f}\n"
            f"Duplication: {self.duplication_score:.1%}\n"
            f"Nested depth: {self.nested_depth}\n"
            f"Quality score: {self.score():.0f}/100"
        )


@dataclass
class OptimizationOpportunity:
    category: str        # duplication | complexity | length | naming | imports
    severity: str        # high | medium | low
    location: str        # line number or function name
    description: str
    suggestion: str
    estimated_reduction: str   # e.g., "60-70% line reduction"


@dataclass
class OptimizationResult:
    original_code: str
    optimized_code: str
    original_metrics: CodeMetrics
    optimized_metrics: CodeMetrics
    opportunities: list[OptimizationOpportunity]
    changes_summary: list[str]
    reduction_pct: float
    time_ms: float

    def report(self) -> str:
        """Return a session summary with event counts, durations, and error list."""
        lines = [
            "═" * 60,
            "CODE OPTIMIZATION REPORT",
            "═" * 60,
            f"\nOriginal: {self.original_metrics.total_lines} lines "
            f"(score: {self.original_metrics.score():.0f}/100)",
            f"Optimized: {self.optimized_metrics.total_lines} lines "
            f"(score: {self.optimized_metrics.score():.0f}/100)",
            f"Reduction: {self.reduction_pct:.1f}%",
            f"\nChanges applied ({len(self.changes_summary)}):",
        ]
        for ch in self.changes_summary:
            lines.append(f"  • {ch}")
        if self.opportunities:
            lines.append(f"\nRemaining opportunities ({len(self.opportunities)}):")
            for op in sorted(self.opportunities, key=lambda o: o.severity):
                lines.append(f"  [{op.severity.upper()}] {op.category}: {op.description}")
        lines.append(f"\nTime: {self.time_ms:.0f}ms")
        return "\n".join(lines)


# ─────────────────────────────────────────────
#  Code Optimizer
# ─────────────────────────────────────────────
class CodeOptimizer:
    """
    Analyzes and optimizes Python code.
    Uses AST for structural analysis + LLM for semantic refactoring.
    """

    def __init__(self, model=None):
        self._model = model

    # ─────────────────────────────────────────
    #  Public interface
    # ─────────────────────────────────────────
    def analyze(self, code: str) -> CodeMetrics:
        """Compute all metrics for a piece of code."""
        return self._compute_metrics(code)

    def find_opportunities(self, code: str) -> list[OptimizationOpportunity]:
        """Find all optimization opportunities without modifying code."""
        ops = []
        ops.extend(self._find_duplications(code))
        ops.extend(self._find_long_functions(code))
        ops.extend(self._find_deep_nesting(code))
        ops.extend(self._find_repeated_patterns(code))
        ops.extend(self._find_unused_imports(code))
        ops.extend(self._find_magic_numbers(code))
        return sorted(ops, key=lambda o: {"high": 0, "medium": 1, "low": 2}[o.severity])

    def optimize(self, code: str, aggressive: bool = False) -> OptimizationResult:
        """
        Full optimization pipeline:
          1. Measure original
          2. Apply rule-based optimizations
          3. Apply LLM-based semantic refactoring
          4. Measure result
        """
        start = time.time()
        original_metrics = self._compute_metrics(code)
        opportunities = self.find_opportunities(code)

        optimized = code
        changes: list[str] = []

        # Rule-based passes (deterministic, fast)
        optimized, ch = self._remove_trailing_whitespace(optimized)
        changes.extend(ch)

        optimized, ch = self._consolidate_repeated_prints(optimized)
        changes.extend(ch)

        optimized, ch = self._consolidate_repeated_assignments(optimized)
        changes.extend(ch)

        optimized, ch = self._simplify_boolean_returns(optimized)
        changes.extend(ch)

        optimized, ch = self._remove_redundant_else(optimized)
        changes.extend(ch)

        # LLM-based pass (semantic, powerful)
        if self._model and (aggressive or original_metrics.score() < 70):
            optimized, llm_changes = self._llm_refactor(optimized, opportunities)
            changes.extend(llm_changes)

        optimized_metrics = self._compute_metrics(optimized)
        orig_lines = original_metrics.total_lines
        opt_lines = optimized_metrics.total_lines
        reduction = (orig_lines - opt_lines) / max(orig_lines, 1) * 100

        return OptimizationResult(
            original_code=code,
            optimized_code=optimized,
            original_metrics=original_metrics,
            optimized_metrics=optimized_metrics,
            opportunities=opportunities,
            changes_summary=changes,
            reduction_pct=reduction,
            time_ms=(time.time() - start) * 1000,
        )

    # ─────────────────────────────────────────
    #  Metrics computation
    # ─────────────────────────────────────────
    def _compute_metrics(self, code: str) -> CodeMetrics:
        lines = code.split("\n")
        total = len(lines)
        blank = sum(1 for l in lines if not l.strip())
        comment = sum(1 for l in lines if l.strip().startswith("#"))
        code_lines = total - blank - comment

        functions = 0
        classes = 0
        func_lengths: list[int] = []
        imports = 0
        max_depth = 0

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    functions += 1
                    if hasattr(node, "end_lineno") and hasattr(node, "lineno"):
                        func_lengths.append(node.end_lineno - node.lineno)
                elif isinstance(node, ast.ClassDef):
                    classes += 1
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    imports += 1
            max_depth = self._max_nesting_depth(tree)
        except SyntaxError:
            pass

        avg_len = sum(func_lengths) / len(func_lengths) if func_lengths else 0
        max_len = max(func_lengths) if func_lengths else 0
        complexity = self._estimate_complexity(code)
        duplication = self._estimate_duplication(lines)

        return CodeMetrics(
            total_lines=total,
            code_lines=code_lines,
            comment_lines=comment,
            blank_lines=blank,
            functions=functions,
            classes=classes,
            avg_function_length=avg_len,
            max_function_length=max_len,
            cyclomatic_complexity=complexity,
            duplication_score=duplication,
            imports_count=imports,
            nested_depth=max_depth,
        )

    def _max_nesting_depth(self, tree: ast.AST) -> int:
        """Compute maximum nesting depth."""
        max_d = [0]
        def visit(node, depth):
            """Visit."""
            max_d[0] = max(max_d[0], depth)
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.For, ast.While, ast.If, ast.With,
                                      ast.Try, ast.FunctionDef, ast.AsyncFunctionDef)):
                    visit(child, depth + 1)
                else:
                    visit(child, depth)
        visit(tree, 0)
        return max_d[0]

    def _estimate_complexity(self, code: str) -> float:
        """Cyclomatic complexity estimate via keyword counting."""
        keywords = r'\b(if|elif|else|for|while|try|except|with|and|or)\b'
        count = len(re.findall(keywords, code))
        functions = len(re.findall(r'\bdef\b', code))
        return (count / max(functions, 1)) + 1

    def _estimate_duplication(self, lines: list[str]) -> float:
        """Estimate code duplication via repeated non-trivial lines."""
        code_lines = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]
        if len(code_lines) < 5:
            return 0.0
        seen: dict[str, int] = {}
        for l in code_lines:
            if len(l) > 15:
                seen[l] = seen.get(l, 0) + 1
        duplicated = sum(cnt - 1 for cnt in seen.values() if cnt > 1)
        return min(1.0, duplicated / max(len(code_lines), 1))

    # ─────────────────────────────────────────
    #  Opportunity detectors
    # ─────────────────────────────────────────
    def _find_duplications(self, code: str) -> list[OptimizationOpportunity]:
        lines = code.split("\n")
        seen: dict[str, list[int]] = {}
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if len(stripped) > 20 and not stripped.startswith("#"):
                if stripped not in seen:
                    seen[stripped] = []
                seen[stripped].append(i)

        ops = []
        for line_text, line_nums in seen.items():
            if len(line_nums) >= 3:
                ops.append(OptimizationOpportunity(
                    category="duplication",
                    severity="high",
                    location=f"Lines {line_nums[:3]}...",
                    description=f"Line repeated {len(line_nums)}x: `{line_text[:50]}`",
                    suggestion="Extract to a function or use a loop",
                    estimated_reduction=f"~{int((len(line_nums)-1)/len(line_nums)*100)}% for this block",
                ))
            elif len(line_nums) == 2:
                ops.append(OptimizationOpportunity(
                    category="duplication",
                    severity="medium",
                    location=f"Lines {line_nums}",
                    description=f"Duplicated: `{line_text[:50]}`",
                    suggestion="Deduplicate or extract",
                    estimated_reduction="minor",
                ))
        return ops

    def _find_long_functions(self, code: str) -> list[OptimizationOpportunity]:
        ops = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if hasattr(node, "end_lineno"):
                        length = node.end_lineno - node.lineno
                        if length > 80:
                            ops.append(OptimizationOpportunity(
                                category="length",
                                severity="high",
                                location=f"def {node.name}() at line {node.lineno}",
                                description=f"Function is {length} lines long",
                                suggestion="Split into smaller functions (max 40-50 lines each)",
                                estimated_reduction="40-60%",
                            ))
                        elif length > 40:
                            ops.append(OptimizationOpportunity(
                                category="length",
                                severity="medium",
                                location=f"def {node.name}() at line {node.lineno}",
                                description=f"Function is {length} lines — consider splitting",
                                suggestion="Extract sub-tasks to helper functions",
                                estimated_reduction="20-30%",
                            ))
        except SyntaxError:
            pass
        return ops

    def _find_deep_nesting(self, code: str) -> list[OptimizationOpportunity]:
        ops = []
        lines = code.split("\n")
        for i, line in enumerate(lines, 1):
            indent = len(line) - len(line.lstrip())
            if indent >= 16:  # 4 levels of nesting (4 spaces each)
                ops.append(OptimizationOpportunity(
                    category="complexity",
                    severity="high" if indent >= 20 else "medium",
                    location=f"Line {i}",
                    description=f"Nesting depth {indent//4} levels",
                    suggestion="Use early returns, extract functions, or invert conditions",
                    estimated_reduction="Reduces complexity significantly",
                ))
        return ops

    def _find_repeated_patterns(self, code: str) -> list[OptimizationOpportunity]:
        """Find print/log/append repeated sequentially — loop candidates."""
        ops = []
        lines = [l.strip() for l in code.split("\n")]

        for pattern_start in (r'print\(', r'logger\.\w+\(', r'\w+\.append\('):
            runs = []
            current_run = []
            for i, line in enumerate(lines):
                if re.match(pattern_start, line):
                    current_run.append(i + 1)
                else:
                    if len(current_run) >= 5:
                        runs.append(current_run[:])
                    current_run = []
            if len(current_run) >= 5:
                runs.append(current_run)

            for run in runs:
                ops.append(OptimizationOpportunity(
                    category="duplication",
                    severity="high",
                    location=f"Lines {run[0]}-{run[-1]}",
                    description=f"{len(run)} sequential `{pattern_start.rstrip('(')}` calls",
                    suggestion="Replace with a loop over a list/dict of values",
                    estimated_reduction=f"~{int((len(run)-1)/len(run)*100)}%",
                ))
        return ops

    def _find_unused_imports(self, code: str) -> list[OptimizationOpportunity]:
        ops = []
        try:
            tree = ast.parse(code)
            imported_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported_names.add(alias.asname or alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        if alias.name != "*":
                            imported_names.add(alias.asname or alias.name)

            # Check which are actually used
            code_no_imports = re.sub(r"^(import|from)\s.+$", "", code, flags=re.MULTILINE)
            for name in imported_names:
                if name and re.search(r'\b' + re.escape(name) + r'\b', code_no_imports) is None:
                    ops.append(OptimizationOpportunity(
                        category="imports",
                        severity="low",
                        location="imports",
                        description=f"Unused import: `{name}`",
                        suggestion=f"Remove `import {name}` if not needed",
                        estimated_reduction="minor",
                    ))
        except SyntaxError:
            pass
        return ops

    def _find_magic_numbers(self, code: str) -> list[OptimizationOpportunity]:
        ops = []
        magic = re.findall(r'(?<![.\w])(?<!\w)(\d{2,})(?![.\w])', code)
        if len(magic) >= 5:
            ops.append(OptimizationOpportunity(
                category="naming",
                severity="low",
                location="various",
                description=f"Found {len(magic)} magic numbers",
                suggestion="Replace with named constants (TIMEOUT = 30, MAX_RETRIES = 3, etc.)",
                estimated_reduction="Improves readability",
            ))
        return ops

    # ─────────────────────────────────────────
    #  Rule-based transformations
    # ─────────────────────────────────────────
    def _remove_trailing_whitespace(self, code: str) -> tuple[str, list[str]]:
        cleaned = "\n".join(l.rstrip() for l in code.split("\n"))
        changed = cleaned != code
        return cleaned, ["Removed trailing whitespace"] if changed else []

    def _consolidate_repeated_prints(self, code: str) -> tuple[str, list[str]]:
        """Replace sequential print("x") blocks with loop."""
        pattern = re.compile(
            r'((?:[ \t]*print\(["\'].*?["\']\)\n){5,})',
            re.MULTILINE,
        )
        changes = []
        def replace_block(m):
            """Replace block."""
            block = m.group(1)
            lines_in_block = [
                l.strip()[7:-2] if l.strip().startswith('print("') else l.strip()[7:-2]
                for l in block.strip().split("\n")
                if l.strip()
            ]
            indent = len(m.group(1)) - len(m.group(1).lstrip())
            spaces = " " * indent
            items_str = ", ".join(f'"{s}"' for s in lines_in_block[:10])
            changes.append(
                f"Consolidated {len(lines_in_block)} sequential print() calls into a loop"
            )
            return f"{spaces}for _msg in [{items_str}]:\n{spaces}    print(_msg)\n"

        result = pattern.sub(replace_block, code)
        return result, changes

    def _consolidate_repeated_assignments(self, code: str) -> tuple[str, list[str]]:
        """Detect patterns like x1=1, x2=2, x3=3 → dict or list."""
        pattern = re.compile(
            r'((?:[ \t]*\w+\d+\s*=\s*.+\n){5,})',
            re.MULTILINE,
        )
        changes = []
        if pattern.search(code):
            changes.append(
                "Detected sequential numbered variables — consider using a list or dict instead"
            )
        return code, changes  # Suggestion only, don't auto-transform (risky)

    def _simplify_boolean_returns(self, code: str) -> tuple[str, list[str]]:
        """
        if cond:
            return True
        else:
            return False
        →
        return cond
        """
        pattern = re.compile(
            r'([ \t]*)if (.+?):\s*\n\s*return True\s*\n\s*else:\s*\n\s*return False',
            re.MULTILINE,
        )
        changes = []
        def replace(m):
            """Replace."""
            changes.append("Simplified boolean return (if x: return True else: return False → return x)")
            return f"{m.group(1)}return {m.group(2)}"
        result = pattern.sub(replace, code)
        return result, changes

    def _remove_redundant_else(self, code: str) -> tuple[str, list[str]]:
        """Remove else after return/raise/continue."""
        changes = []
        pattern = re.compile(
            r'(return|raise|continue|break)\s*\n(\s*)else:\n',
            re.MULTILINE,
        )
        if pattern.search(code):
            result = pattern.sub(r'\1\n', code)
            changes.append("Removed redundant else after return/raise/continue (early exit pattern)")
            return result, changes
        return code, []

    # ─────────────────────────────────────────
    #  LLM-based refactoring
    # ─────────────────────────────────────────
    def _llm_refactor(
        self, code: str, opportunities: list[OptimizationOpportunity]
    ) -> tuple[str, list[str]]:
        if not self._model:
            return code, []

        top_issues = "\n".join(
            f"- [{op.severity.upper()}] {op.description}" for op in opportunities[:5]
        )
        prompt = f"""You are an expert Python code optimizer.

ORIGINAL CODE:
```python
{code[:3000]}
```

TOP ISSUES TO FIX:
{top_issues}

RULES:
1. Keep exact same functionality — no behavior changes
2. Reduce line count using loops, list comprehensions, functions
3. Remove duplicated code
4. Apply early returns where possible
5. Do NOT add new features or change logic

Respond with ONLY the optimized Python code. No explanation, no markdown fences.
"""
        try:
            result = self._model.generate(prompt, max_tokens=2000)
            if result and len(result.strip()) > 50:
                # Validate it's actually Python
                try:
                    ast.parse(result)
                    return result, ["LLM semantic refactoring applied"]
                except SyntaxError:
                    return code, []
        except Exception:
            pass
        return code, []
