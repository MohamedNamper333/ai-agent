import asyncio
import ast
import datetime
import json
import math
import re
import subprocess
import sys
import time
import hashlib
from pathlib import Path
from typing import Any, Optional
from dataclasses import dataclass, field

import config


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    result: str
    error: str = ""
    execution_time: float = 0.0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%H:%M:%S")


class Tool:
    def __init__(self, name: str, description: str, func: callable,
                 timeout: int = 30, requires_auth: bool = False,
                 category: str = "general"):
        self.name = name
        self.description = description
        self.func = func
        self.timeout = timeout
        self.requires_auth = requires_auth
        self.category = category
        self._call_count = 0
        self._error_count = 0
        self._total_time = 0.0

    def run(self, **kwargs) -> ToolResult:
        start_time = time.time()
        self._call_count += 1

        try:
            result = self.func(**kwargs)
            execution_time = time.time() - start_time
            self._total_time += execution_time
            return ToolResult(
                tool_name=self.name,
                success=True,
                result=str(result),
                execution_time=execution_time,
            )
        except Exception as e:
            execution_time = time.time() - start_time
            self._total_time += execution_time
            self._error_count += 1
            return ToolResult(
                tool_name=self.name,
                success=False,
                result="",
                error=str(e),
                execution_time=execution_time,
            )

    def get_stats(self) -> dict:
        avg_time = self._total_time / max(self._call_count, 1)
        error_rate = self._error_count / max(self._call_count, 1) * 100
        return {
            "name": self.name,
            "calls": self._call_count,
            "errors": self._error_count,
            "error_rate": f"{error_rate:.1f}%",
            "avg_time": f"{avg_time:.3f}s",
            "total_time": f"{self._total_time:.3f}s",
        }


class ToolRegistry:
    def __init__(self, agent: Any = None):
        self._tools: dict[str, Tool] = {}
        self._agent = agent
        self._execution_log: list[dict] = []
        self._parallel_enabled = True
        self._enabled: set[str] = set()
        self._lazy_loaders: dict[str, callable] = {}
        self._loaded_categories: set[str] = set()
        self._register_defaults()
        self._register_lazy_categories()
        self._apply_enabled_filter()
        self._list_cache: list[Tool] | None = None
        self._format_prompt_cache: str | None = None

    def _register_lazy_categories(self):
        category_map = {
            "file": (self._register_file_tools, 9),
            "web": (self._register_web_tools, 3),
            "git": (self._register_git_tools, 8),
            "code": (self._register_code_tools, 9),
            "data": (self._register_data_tools, 10),
            "memory": (self._register_long_term_memory, 2),
            "multi_agent": (self._register_multi_agent, 2),
            "documents": (self._register_document_tools, 6),
            "voice": (self._register_voice_tools, 3),
            "scheduler": (self._register_scheduler, 3),
            "docker": (self._register_docker_tools, 2),
            "self_improve": (self._register_self_improve, 4),
        }
        self._lazy_tool_counts = {}
        for cat, (loader, count) in category_map.items():
            self._lazy_loaders[cat] = loader
            self._lazy_tool_counts[cat] = count
            self._loaded_categories.add("basic")

    def _ensure_category(self, category: str):
        if category in self._lazy_loaders and category not in self._loaded_categories:
            self._lazy_loaders[category]()
            self._loaded_categories.add(category)

    def _apply_enabled_filter(self):
        if config.TOOLS_ENABLED:
            self._enabled = set(config.TOOLS_ENABLED)
        else:
            self._enabled = set(self._tools.keys())

    def enable_tool(self, name: str) -> bool:
        if name in self._tools:
            self._enabled.add(name)
            self._invalidate_cache()
            return True
        return False

    def disable_tool(self, name: str) -> bool:
        if name in self._tools:
            self._enabled.discard(name)
            self._invalidate_cache()
            return True
        return False

    def enable_category(self, category: str) -> int:
        count = 0
        for tool in self._tools.values():
            if tool.category == category:
                self._enabled.add(tool.name)
                count += 1
        if count:
            self._invalidate_cache()
        return count

    def disable_category(self, category: str) -> int:
        count = 0
        for tool in self._tools.values():
            if tool.category == category:
                self._enabled.discard(tool.name)
                count += 1
        if count:
            self._invalidate_cache()
        return count

    def is_enabled(self, name: str) -> bool:
        return name in self._enabled

    def get_enabled_count(self) -> int:
        self._ensure_all()
        return len(self._enabled)

    def get_disabled_count(self) -> int:
        return self.total_count() - len(self._enabled)

    def total_count(self) -> int:
        self._ensure_all()
        return len(self._tools)

    def get_enabled_tools(self) -> list[Tool]:
        return [t for t in self._tools.values() if t.name in self._enabled]

    def get_disabled_tools(self) -> list[Tool]:
        return [t for t in self._tools.values() if t.name not in self._enabled]

    def _register_defaults(self):
        self.register(Tool(
            "datetime",
            "Get the current date and time",
            lambda: datetime.datetime.now().isoformat(),
            category="basic",
        ))

        self.register(Tool(
            "calculator",
            "Evaluate a mathematical expression. Input: a math expression string",
            self._safe_calculate,
            category="basic",
        ))

        self.register(Tool(
            "run_code",
            "Execute Python code and return the output. Input: Python code as string",
            self._run_python,
            timeout=60,
            category="code",
        ))

        self.register(Tool(
            "search_memory",
            "Search past conversations. Input: query string",
            lambda query: f"[memory search for: {query}]",
            category="memory",
        ))

    def _safe_calculate(self, expr: str) -> str:
        if re.search(r'(__import__|exec|eval|open|os\.|sys\.|subprocess|import|lambda|class|def)', expr):
            return "Error: Forbidden expression"
        try:
            tree = ast.parse(expr, mode='eval')
            return str(self._eval_ast_node(tree.body))
        except Exception as e:
            return f"Error: {e}"

    def _eval_ast_node(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = self._eval_ast_node(node.left)
            right = self._eval_ast_node(node.right)
            if isinstance(node.op, ast.Add): return left + right
            elif isinstance(node.op, ast.Sub): return left - right
            elif isinstance(node.op, ast.Mult): return left * right
            elif isinstance(node.op, ast.Div): return left / right
            elif isinstance(node.op, ast.FloorDiv): return left // right
            elif isinstance(node.op, ast.Mod): return left % right
            elif isinstance(node.op, ast.Pow): return left ** right
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_ast_node(node.operand)
            if isinstance(node.op, ast.UAdd): return +operand
            elif isinstance(node.op, ast.USub): return -operand
            raise ValueError(f"Unsupported unary: {type(node.op).__name__}")
        elif isinstance(node, ast.Call):
            func = self._eval_ast_node(node.func)
            args = [self._eval_ast_node(a) for a in node.args]
            if callable(func):
                return func(*args)
            raise ValueError(f"Not callable: {func}")
        elif isinstance(node, ast.Name):
            safe_funcs = {
                'abs': abs, 'round': round, 'min': min, 'max': max,
                'sum': sum, 'pow': pow, 'int': int, 'float': float,
                'pi': math.pi, 'e': math.e, 'sqrt': math.sqrt,
                'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
                'log': math.log, 'log10': math.log10, 'log2': math.log2,
            }
            if node.id in safe_funcs:
                return safe_funcs[node.id]
            raise ValueError(f"Unknown name: {node.id}")
        elif isinstance(node, ast.Tuple):
            return tuple(self._eval_ast_node(e) for e in node.elts)
        elif isinstance(node, ast.List):
            return [self._eval_ast_node(e) for e in node.elts]
        raise ValueError(f"Unsupported expression: {type(node).__name__}")

    def _register_file_tools(self):
        from tools.file_ops import FileOps
        fo = FileOps()

        self.register(Tool(
            "read_file",
            "Read a file from the filesystem. Params: path (required), offset (optional, line number), limit (optional, max lines)",
            fo.read_file,
            category="file",
        ))
        self.register(Tool(
            "write_file",
            "Write content to a file (creates directories if needed). Params: path, content",
            fo.write_file,
            category="file",
        ))
        self.register(Tool(
            "edit_file",
            "Find and replace text in a file. Params: path, old_string, new_string",
            fo.edit_file,
            category="file",
        ))
        self.register(Tool(
            "glob",
            "Find files by glob pattern. Params: pattern (like **/*.py), path (optional)",
            fo.glob_search,
            category="file",
        ))
        self.register(Tool(
            "grep",
            "Search file contents by regex. Params: pattern, path (optional), include (file pattern, default: *)",
            fo.grep_search,
            category="file",
        ))
        self.register(Tool(
            "list_dir",
            "List directory contents. Params: path (optional, defaults to current)",
            fo.list_directory,
            category="file",
        ))
        self.register(Tool(
            "file_info",
            "Get file/directory metadata. Params: path",
            fo.file_info,
            category="file",
        ))
        self.register(Tool(
            "file_compare",
            "Compare two files and show differences. Params: file1, file2",
            fo.file_compare,
            category="file",
        ))
        self.register(Tool(
            "batch_read",
            "Read multiple files at once. Params: paths (comma-separated)",
            fo.batch_read,
            category="file",
        ))

    def _register_web_tools(self):
        from tools.web_search import WebTools
        wt = WebTools()

        self.register(Tool(
            "fetch_url",
            "Fetch and extract text content from a URL. Params: url",
            wt.fetch_url,
            timeout=30,
            category="web",
        ))
        self.register(Tool(
            "search_web",
            "Search the internet. Params: query (search term), num_results (optional, default 5)",
            wt.search_web,
            timeout=15,
            category="web",
        ))
        self.register(Tool(
            "web_scrape",
            "Scrape and extract text content from a URL. Params: url, selector (optional CSS selector)",
            wt.web_scrape,
            category="web",
        ))

    def _register_git_tools(self):
        from tools.git_ops import GitOps
        go = GitOps()

        self.register(Tool(
            "git_status",
            "Show git working tree status. Params: path (optional)",
            go.git_status,
            category="git",
        ))
        self.register(Tool(
            "git_diff",
            "Show git diff. Params: path (optional), staged (optional, true/false)",
            go.git_diff,
            category="git",
        ))
        self.register(Tool(
            "git_log",
            "Show git commit log. Params: path (optional), count (optional, default 20)",
            go.git_log,
            category="git",
        ))
        self.register(Tool(
            "git_branch",
            "List git branches. Params: path (optional)",
            go.git_branch,
            category="git",
        ))
        self.register(Tool(
            "git_show",
            "Show a git commit details. Params: commit (default HEAD), path (optional)",
            go.git_show,
            category="git",
        ))
        self.register(Tool(
            "git_add",
            "Add files to git staging. Params: files (comma-separated or . for all), path (optional)",
            go.git_add,
            category="git",
        ))
        self.register(Tool(
            "git_commit",
            "Create a git commit. Params: message, path (optional)",
            go.git_commit,
            category="git",
        ))
        self.register(Tool(
            "git_blame",
            "Show who wrote each line. Params: file_path, path (optional)",
            go.git_blame,
            category="git",
        ))

    def _register_code_tools(self):
        from tools.code_analysis import CodeAnalysis
        ca = CodeAnalysis()

        self.register(Tool(
            "scan_project",
            "Scan a project directory and show its structure. Params: path (optional)",
            ca.scan_project,
            category="code",
        ))
        self.register(Tool(
            "review_code",
            "Review a code file for bugs, issues, and quality. Params: file_path",
            ca.review_code,
            category="code",
        ))
        self.register(Tool(
            "analyze_imports",
            "Analyze imports/dependencies of a Python file. Params: file_path",
            ca.analyze_imports,
            category="code",
        ))
        self.register(Tool(
            "code_refactor",
            "Refactor code for better readability and performance. Params: file_path, instructions",
            ca.code_refactor,
            category="code",
        ))
        self.register(Tool(
            "complexity_metrics",
            "Calculate code complexity metrics. Params: file_path",
            ca.complexity_metrics,
            category="code",
        ))
        self.register(Tool(
            "dependency_graph",
            "Show dependency graph of a project. Params: path (optional)",
            ca.dependency_graph,
            category="code",
        ))
        self.register(Tool(
            "analyze_security",
            "Security vulnerability scan of a code file. Params: file_path",
            ca.analyze_security,
            category="code",
        ))
        self.register(Tool(
            "analyze_code_quality",
            "Detailed code quality report with metrics. Params: file_path",
            ca.analyze_code_quality,
            category="code",
        ))
        self.register(Tool(
            "generate_test",
            "Generate unit test template for a Python file. Params: file_path, function_name (optional)",
            ca.generate_test,
            category="code",
        ))

    def _register_data_tools(self):
        from tools.data_analysis import DataAnalysis
        da = DataAnalysis()

        self.register(Tool(
            "analyze_csv",
            "Analyze a CSV file: columns, types, stats, preview. Params: file_path",
            da.analyze_csv,
            category="data",
        ))
        self.register(Tool(
            "analyze_json",
            "Analyze a JSON file: structure, keys, preview. Params: file_path",
            da.analyze_json,
            category="data",
        ))
        self.register(Tool(
            "analyze_text",
            "Analyze a text file: chars, words, lines, stats. Params: file_path",
            da.analyze_text,
            category="data",
        ))
        self.register(Tool(
            "stats_summary",
            "Compute statistical summary of numeric data. Params: data_json (JSON array of numbers)",
            da.stats_summary,
            category="data",
        ))
        self.register(Tool(
            "analyze_excel",
            "Analyze an Excel file. Params: file_path, sheet (optional)",
            da.analyze_excel,
            category="data",
        ))
        self.register(Tool(
            "sql_query",
            "Run SQL query on CSV/JSON data. Params: file_path, query",
            da.sql_query,
            category="data",
        ))
        self.register(Tool(
            "analyze_data_quality",
            "Full data quality report: missing values, duplicates, outliers. Params: file_path",
            da.analyze_data_quality,
            category="data",
        ))
        self.register(Tool(
            "correlation_analysis",
            "Find correlations between numeric columns. Params: file_path, columns (optional comma-separated)",
            da.correlation_analysis,
            category="data",
        ))
        self.register(Tool(
            "generate_visualization",
            "Generate a chart from data. Params: file_path, chart_type (auto/scatter/histogram/bar/pie/correlation), x_col (optional), y_col (optional)",
            da.generate_visualization,
            category="data",
        ))
        self.register(Tool(
            "time_series_analysis",
            "Analyze time series data for trends and patterns. Params: file_path, date_col (optional), value_col (optional)",
            da.time_series_analysis,
            category="data",
        ))

    def _run_python(self, code: str, timeout: int = 30) -> str:
        import tempfile

        # Multi-layer security
        dangerous_patterns = [
            r'__import__\s*\(',
            r'import\s+(os|sys|subprocess|shutil|socket|pty|builtins|ctypes)',
            r'from\s+(os|sys|subprocess|shutil|socket|pty|builtins|ctypes)',
            r'os\.\s*(system|popen|exec|execl|fork|kill|chmod|chown|remove|unlink|rmdir|makedirs)',
            r'subprocess\.\s*(run|Popen|call|check_call|check_output)',
            r'(?<!\w)eval\s*\(',
            r'(?<!\w)exec\s*\(',
            r'(?<!\w)open\s*\(',
            r'breakpoint\s*\(',
            r'(?<!\w)input\s*\(',
            r'__reduce__',
            r'__subclasses__',
            r'__globals__',
            r'__builtins__',
            r'marshal\.\s*(loads?|dumps)',
            r'pickle\.\s*(loads?|dumps)',
            r'shelve\.\s*open',
            r'compile\s*\(',
            r'\.__dict__',
            r'getattr\s*\(.*__',
            r'setattr\s*\(.*__',
        ]

        restricted_imports = [
            'os', 'sys', 'subprocess', 'shutil', 'socket', 'pty', 'builtins',
            'ctypes', 'signal', 'multiprocessing', 'threading', 'asyncio',
            'importlib', 'pkgutil', 'inspect', 'traceback', 'gc',
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                return f"Error: Code contains restricted pattern"

        for mod in restricted_imports:
            if re.search(rf'(import\s+{mod}|from\s+{mod}\s+)', code, re.IGNORECASE):
                return f"Error: Import of '{mod}' is restricted"

        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
                f.write(code)
                tmp_path = f.name.replace("\\", "/")

            result = subprocess.run(
                [sys.executable, "-c",
                 f"import sys; sys.path = []; "
                 f"sys.modules.pop('os', None); sys.modules.pop('subprocess', None); "
                 f"sys.modules.pop('sys', None); "
                 f"exec(open('{tmp_path}').read())"],
                capture_output=True, text=True, timeout=timeout,
                env={"PYTHONPATH": "", "PATH": "", "HOME": "", "USER": "sandbox"},
            )

            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass

            output = result.stdout or ""
            if result.stderr:
                error_lines = [l for l in result.stderr.split('\n')
                              if 'Warning' not in l and 'warn' not in l.lower()]
                if error_lines:
                    output += f"\nStderr: {''.join(error_lines)[:500]}"
            if result.returncode != 0:
                output += f"\nExit code: {result.returncode}"
            return output.strip() or "(no output)"

        except subprocess.TimeoutExpired:
            return f"Error: Code execution timed out ({timeout}s)"
        except Exception as e:
            return f"Error: {e}"

    def _register_long_term_memory(self):
        from tools.long_term_memory import LongTermMemory
        self._ltm = LongTermMemory()

        self.register(Tool(
            "recall",
            "Search long-term memory for past conversations. Params: query",
            self._ltm.search,
            category="memory",
        ))
        self.register(Tool(
            "remember",
            "Store an important fact in long-term memory. Params: summary, topics (comma-separated)",
            lambda summary, topics="": self._ltm.add_summary("manual", summary, [t.strip() for t in topics.split(",") if t.strip()]) or f"Stored: {summary[:100]}..." if True else "",
            category="memory",
        ))

    def _register_multi_agent(self):
        from tools.multi_agent import MultiAgentOrchestrator
        self._orchestrator = MultiAgentOrchestrator()

        self.register(Tool(
            "council",
            "Convene multiple specialist agents to analyze a problem. Params: task, context (optional)",
            self._orchestrator.run_council,
            category="multi_agent",
        ))
        self.register(Tool(
            "delegate",
            "Delegate a task to a specialist agent (The Analyst, The Programmer, The Reviewer). Params: agent_name, task, context (optional)",
            self._orchestrator.delegate,
            category="multi_agent",
        ))

    def _register_document_tools(self):
        from tools.documents import DocumentTools
        dt = DocumentTools()

        self.register(Tool(
            "read_pdf",
            "Extract text from a PDF file. Params: file_path, max_pages (optional, default 20)",
            dt.read_pdf,
            category="documents",
        ))
        self.register(Tool(
            "read_docx",
            "Extract text from a Word DOCX file. Params: file_path",
            dt.read_docx,
            category="documents",
        ))
        self.register(Tool(
            "analyze_image",
            "Analyze an image file: format, size, colors, OCR. Params: file_path",
            dt.analyze_image,
            category="documents",
        ))
        self.register(Tool(
            "ocr_image",
            "Extract text from an image using OCR. Params: file_path",
            dt.ocr_image,
            category="documents",
        ))
        self.register(Tool(
            "read_excel",
            "Extract data from Excel file. Params: file_path, sheet (optional)",
            dt.read_excel,
            category="documents",
        ))
        self.register(Tool(
            "html_to_text",
            "Convert HTML to clean text. Params: html_content or url",
            dt.html_to_text,
            category="documents",
        ))

    def _register_voice_tools(self):
        from tools.voice import VoiceTools
        vt = VoiceTools()

        self.register(Tool(
            "listen",
            "Listen to microphone and convert speech to text. Params: timeout (optional, default 5s)",
            vt.listen,
            category="voice",
        ))
        self.register(Tool(
            "speak",
            "Convert text to speech and play it. Params: text",
            vt.speak,
            category="voice",
        ))
        self.register(Tool(
            "save_speech",
            "Convert text to speech and save as MP3 file. Params: text, file_path (optional)",
            vt.save_speech,
            category="voice",
        ))

    def _register_scheduler(self):
        from tools.scheduler import TaskScheduler
        self._scheduler = TaskScheduler()
        self._scheduler.load()

        self.register(Tool(
            "schedule_task",
            "Schedule a recurring task for the agent. Params: name, prompt, schedule (minutes or HH:MM), task_type (interval/daily)",
            self._scheduler.add_task,
            category="scheduler",
        ))
        self.register(Tool(
            "list_scheduled_tasks",
            "List all scheduled tasks",
            lambda: "\n".join(f"{t['id']}: {t['name']} ({t['schedule']}) - enabled={t['enabled']}" for t in self._scheduler.list_tasks()) or "No tasks scheduled",
            category="scheduler",
        ))
        self.register(Tool(
            "remove_scheduled_task",
            "Remove a scheduled task. Params: task_id",
            self._scheduler.remove_task,
            category="scheduler",
        ))

    def _register_docker_tools(self):
        from tools.docker_sandbox import DockerSandbox
        ds = DockerSandbox()

        self.register(Tool(
            "docker_run",
            "Execute code in an isolated Docker container (secure). Params: code, language (python/bash/node, default python), timeout (default 30)",
            ds.run_code,
            timeout=60,
            category="docker",
        ))
        self.register(Tool(
            "docker_images",
            "List available Docker images",
            ds.list_images,
            category="docker",
        ))

    def _register_self_improve(self):
        from tools.self_improve import SelfImprover
        self._improver = SelfImprover()

        self.register(Tool(
            "self_analyze",
            "Analyze the agent's own codebase structure",
            self._improver.analyze_codebase,
            category="self_improve",
        ))
        self.register(Tool(
            "self_review",
            "Run a full code review of the agent's own code and suggest improvements",
            self._improver.run_self_review,
            category="self_improve",
        ))
        self.register(Tool(
            "suggest_improvements",
            "Suggest specific improvements for a code file. Params: file_path",
            self._improver.suggest_improvements,
            category="self_improve",
        ))
        self.register(Tool(
            "apply_improvement",
            "Apply a code improvement to a file. Params: file_path, instructions",
            self._improver.apply_improvement,
            category="self_improve",
        ))

    def _invalidate_cache(self) -> None:
        self._list_cache = None
        self._format_prompt_cache = None

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        self._enabled.add(tool.name)
        self._invalidate_cache()

    def get(self, name: str) -> Tool | None:
        for cat in list(self._lazy_loaders.keys()):
            if cat not in self._loaded_categories:
                self._ensure_category(cat)
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        if self._list_cache is not None:
            return self._list_cache
        self._ensure_all()
        result = [t for t in self._tools.values() if t.name in self._enabled]
        self._list_cache = result
        return result

    def list_all_tools(self) -> list[Tool]:
        self._ensure_all()
        return list(self._tools.values())

    def list_tools_by_category(self) -> dict[str, list[Tool]]:
        self._ensure_all()
        categories = {}
        for tool in self._tools.values():
            if tool.name not in self._enabled:
                continue
            cat = tool.category
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(tool)
        return categories

    def list_tools_by_category_all(self) -> dict[str, list[Tool]]:
        self._ensure_all()
        categories = {}
        for tool in self._tools.values():
            cat = tool.category
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(tool)
        return categories

    def _ensure_all(self):
        for cat in list(self._lazy_loaders.keys()):
            if cat not in self._loaded_categories:
                self._ensure_category(cat)

    def format_for_prompt(self) -> str:
        if self._format_prompt_cache is not None:
            return self._format_prompt_cache
        if not self._tools:
            return ""
        total = len(self._tools)
        enabled = len(self._enabled)
        lines = [f"## Available Tools ({enabled}/{total} tools enabled)", ""]
        categories = self.list_tools_by_category()

        category_labels = {
            "basic": "Basic",
            "file": "File Operations",
            "web": "Web",
            "git": "Git",
            "code": "Code Analysis & Security",
            "data": "Data Analysis & Visualization",
            "documents": "Documents & Images",
            "voice": "Voice",
            "multi_agent": "Multi-Agent",
            "scheduler": "Scheduling",
            "docker": "Docker Sandbox",
            "self_improve": "Self-Improvement",
            "memory": "Memory",
            "general": "General",
        }

        for cat, tools in categories.items():
            label = category_labels.get(cat, cat.title())
            lines.append(f"### {label}")
            for t in tools:
                lines.append(f"- `{t.name}`: {t.description[:120]}")
            lines.append("")

        lines.append("## How to use tools")
        lines.append(
            'To use a tool, respond with: {"tool_calls": [{"name": "tool_name", "arguments": {"param": "value"}}]}\n'
            "Or for legacy format: <tool name=\"tool_name\">param=value</tool>\n"
            "Then wait for the result before continuing."
        )
        result = "\n".join(lines)
        self._format_prompt_cache = result
        return result

    def parse_and_execute(self, text: str) -> tuple[list[dict], list[dict]]:
        calls = []
        results = []

        json_calls = self._parse_json_tool_calls(text)
        if json_calls:
            calls = json_calls
            for call in json_calls:
                tool = self.get(call["name"])
                if not tool:
                    results.append({
                        "tool": call["name"],
                        "result": f"Unknown tool: {call['name']}",
                        "error": f"Unknown tool: {call['name']}",
                        "success": False,
                    })
                    continue
                if tool.name not in self._enabled:
                    results.append({
                        "tool": call["name"],
                        "result": f"Tool '{call['name']}' is disabled",
                        "error": f"Tool '{call['name']}' is disabled",
                        "success": False,
                    })
                    continue

                result = tool.run(**call.get("arguments", {}))
                results.append({
                    "tool": call["name"],
                    "result": result.result,
                    "error": result.error,
                    "success": result.success,
                    "execution_time": result.execution_time,
                })
            return calls, results

        pattern = r'<tool name="([^"]+)">(.*?)</tool>'
        for match in re.finditer(pattern, text, re.DOTALL):
            name = match.group(1)
            params_str = match.group(2).strip()
            calls.append({"name": name, "arguments": {"params_str": params_str}}) # simplified for legacy
            
            tool = self.get(name)
            if not tool:
                results.append({"tool": name, "result": f"Unknown tool: {name}", "success": False})
                continue

            kwargs = {}
            if params_str:
                for pair in params_str.split("|"):
                    pair = pair.strip()
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        kwargs[k.strip()] = v.strip()

            result = tool.run(**kwargs)
            results.append({
                "tool": name,
                "result": result.result,
                "error": result.error,
                "success": result.success,
            })
        
        return calls, results

    def _parse_json_tool_calls(self, text: str) -> list[dict]:
        calls = []

        patterns = [
            r'\{[^{}]*"tool_calls"\s*:\s*\[[\s\S]*?\]\s*\}[^{}]*',
            r'"tool_calls"\s*:\s*\[(.*?)\]',
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text, re.DOTALL):
                try:
                    json_str = match.group()
                    if not json_str.startswith("{"):
                        json_str = "{" + json_str + "}"
                    data = json.loads(json_str)
                    if "tool_calls" in data:
                        for tc in data["tool_calls"]:
                            if isinstance(tc, dict) and "name" in tc:
                                calls.append({
                                    "name": tc["name"],
                                    "arguments": tc.get("arguments", {}),
                                })
                        return calls
                except (json.JSONDecodeError, KeyError):
                    continue

        single_pattern = r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{[^}]*\})\s*\}'
        for match in re.finditer(single_pattern, text):
            try:
                name = match.group(1)
                args = json.loads(match.group(2))
                calls.append({"name": name, "arguments": args})
            except (json.JSONDecodeError, KeyError):
                continue

        return calls

    def contains_tool_call(self, text: str) -> bool:
        if '"tool_calls"' in text and '"name"' in text:
            return True
        return '<tool name="' in text and "</tool>" in text

    async def execute_parallel(self, tool_calls: list[dict]) -> list[ToolResult]:
        results = []

        async def run_single(call: dict) -> ToolResult:
            tool = self.get(call["name"])
            if not tool:
                return ToolResult(
                    tool_name=call["name"],
                    success=False,
                    result="",
                    error=f"Unknown tool: {call['name']}",
                )

            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: tool.run(**call.get("arguments", {})))

        tasks = [run_single(tc) for tc in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(ToolResult(
                    tool_name=tool_calls[i]["name"],
                    success=False,
                    result="",
                    error=str(result),
                ))
            else:
                final_results.append(result)

        return final_results

    def get_tool_stats(self) -> list[dict]:
        stats = []
        for tool in self._tools.values():
            stats.append(tool.get_stats())
        return sorted(stats, key=lambda x: x["calls"], reverse=True)

    def get_registry_stats(self) -> dict:
        total_calls = sum(t._call_count for t in self._tools.values())
        total_errors = sum(t._error_count for t in self._tools.values())
        return {
            "total_tools": len(self._tools),
            "total_calls": total_calls,
            "total_errors": total_errors,
            "error_rate": f"{total_errors / max(total_calls, 1) * 100:.1f}%",
            "categories": len(self.list_tools_by_category()),
        }
