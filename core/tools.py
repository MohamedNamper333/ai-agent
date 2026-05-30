import datetime
import json
import re
import subprocess
import sys
from typing import Any


class Tool:
    def __init__(self, name: str, description: str, func: callable):
        self.name = name
        self.description = description
        self.func = func

    def run(self, **kwargs) -> str:
        try:
            result = self.func(**kwargs)
            return str(result)
        except Exception as e:
            return f"Error: {e}"


class ToolRegistry:
    def __init__(self, agent: Any = None):
        self._tools: dict[str, Tool] = {}
        self._agent = agent
        self._register_defaults()
        self._register_file_tools()
        self._register_web_tools()
        self._register_git_tools()
        self._register_code_tools()
        self._register_data_tools()
        self._register_long_term_memory()
        self._register_multi_agent()
        self._register_document_tools()
        self._register_voice_tools()
        self._register_scheduler()
        self._register_docker_tools()
        self._register_self_improve()

    def _register_defaults(self):
        self.register(Tool(
            "datetime",
            "Get the current date and time",
            lambda: datetime.datetime.now().isoformat(),
        ))

        self.register(Tool(
            "calculator",
            "Evaluate a mathematical expression. Input: a math expression string",
            lambda expr: str(eval(expr, {"__builtins__": {}}, {"abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow})),
        ))

        self.register(Tool(
            "run_code",
            "Execute Python code and return the output. Input: Python code as string",
            self._run_python,
        ))

        self.register(Tool(
            "search_memory",
            "Search past conversations. Input: query string",
            lambda query: f"[memory search for: {query}]",
        ))

    def _register_file_tools(self):
        from tools.file_ops import FileOps
        fo = FileOps()

        self.register(Tool(
            "read_file",
            "Read a file from the filesystem. Params: path (required), offset (optional, line number), limit (optional, max lines)",
            fo.read_file,
        ))
        self.register(Tool(
            "write_file",
            "Write content to a file (creates directories if needed). Params: path, content",
            fo.write_file,
        ))
        self.register(Tool(
            "edit_file",
            "Find and replace text in a file. Params: path, old_string, new_string",
            fo.edit_file,
        ))
        self.register(Tool(
            "glob",
            "Find files by glob pattern. Params: pattern (like **/*.py), path (optional)",
            fo.glob_search,
        ))
        self.register(Tool(
            "grep",
            "Search file contents by regex. Params: pattern, path (optional), include (file pattern, default: *)",
            fo.grep_search,
        ))
        self.register(Tool(
            "list_dir",
            "List directory contents. Params: path (optional, defaults to current)",
            fo.list_directory,
        ))
        self.register(Tool(
            "file_info",
            "Get file/directory metadata. Params: path",
            fo.file_info,
        ))

    def _register_web_tools(self):
        from tools.web_search import WebTools
        wt = WebTools()

        self.register(Tool(
            "fetch_url",
            "Fetch and extract text content from a URL. Params: url",
            wt.fetch_url,
        ))
        self.register(Tool(
            "search_web",
            "Search the internet. Params: query (search term), num_results (optional, default 5)",
            wt.search_web,
        ))

    def _register_git_tools(self):
        from tools.git_ops import GitOps
        go = GitOps()

        self.register(Tool(
            "git_status",
            "Show git working tree status. Params: path (optional)",
            go.git_status,
        ))
        self.register(Tool(
            "git_diff",
            "Show git diff. Params: path (optional), staged (optional, true/false)",
            go.git_diff,
        ))
        self.register(Tool(
            "git_log",
            "Show git commit log. Params: path (optional), count (optional, default 20)",
            go.git_log,
        ))
        self.register(Tool(
            "git_branch",
            "List git branches. Params: path (optional)",
            go.git_branch,
        ))
        self.register(Tool(
            "git_show",
            "Show a git commit details. Params: commit (default HEAD), path (optional)",
            go.git_show,
        ))

    def _register_code_tools(self):
        from tools.code_analysis import CodeAnalysis
        ca = CodeAnalysis()

        self.register(Tool(
            "scan_project",
            "Scan a project directory and show its structure. Params: path (optional)",
            ca.scan_project,
        ))
        self.register(Tool(
            "review_code",
            "Review a code file for bugs, issues, and quality. Params: file_path",
            ca.review_code,
        ))
        self.register(Tool(
            "analyze_imports",
            "Analyze imports/dependencies of a Python file. Params: file_path",
            ca.analyze_imports,
        ))

    def _register_data_tools(self):
        from tools.data_analysis import DataAnalysis
        da = DataAnalysis()

        self.register(Tool(
            "analyze_csv",
            "Analyze a CSV file: columns, types, stats, preview. Params: file_path",
            da.analyze_csv,
        ))
        self.register(Tool(
            "analyze_json",
            "Analyze a JSON file: structure, keys, preview. Params: file_path",
            da.analyze_json,
        ))
        self.register(Tool(
            "analyze_text",
            "Analyze a text file: chars, words, lines, stats. Params: file_path",
            da.analyze_text,
        ))
        self.register(Tool(
            "stats_summary",
            "Compute statistical summary of numeric data. Params: data_json (JSON array of numbers)",
            da.stats_summary,
        ))

    def _run_python(self, code: str) -> str:
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=30,
            )
            output = result.stdout or ""
            if result.returncode != 0:
                output += f"\nExit code: {result.returncode}"
            if result.stderr:
                output += f"\nStderr: {result.stderr[:500]}"
            return output.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: Code execution timed out (30s)"
        except Exception as e:
            return f"Error: {e}"

    def _register_long_term_memory(self):
        from tools.long_term_memory import LongTermMemory
        self._ltm = LongTermMemory()

        self.register(Tool(
            "recall",
            "Search long-term memory for past conversations. Params: query",
            self._ltm.search,
        ))
        self.register(Tool(
            "remember",
            "Store an important fact in long-term memory. Params: summary, topics (comma-separated)",
            lambda summary, topics="": self._ltm.add_summary("manual", summary, [t.strip() for t in topics.split(",") if t.strip()]) or f"Stored: {summary[:100]}..." if True else "",
        ))

    def _register_multi_agent(self):
        from tools.multi_agent import MultiAgentOrchestrator
        self._orchestrator = MultiAgentOrchestrator()

        self.register(Tool(
            "council",
            "Convene multiple specialist agents to analyze a problem. Params: task, context (optional)",
            self._orchestrator.run_council,
        ))
        self.register(Tool(
            "delegate",
            "Delegate a task to a specialist agent (The Analyst, The Programmer, The Reviewer). Params: agent_name, task, context (optional)",
            self._orchestrator.delegate,
        ))

    def _register_document_tools(self):
        from tools.documents import DocumentTools
        dt = DocumentTools()

        self.register(Tool(
            "read_pdf",
            "Extract text from a PDF file. Params: file_path, max_pages (optional, default 20)",
            dt.read_pdf,
        ))
        self.register(Tool(
            "read_docx",
            "Extract text from a Word DOCX file. Params: file_path",
            dt.read_docx,
        ))
        self.register(Tool(
            "analyze_image",
            "Analyze an image file: format, size, colors, OCR. Params: file_path",
            dt.analyze_image,
        ))
        self.register(Tool(
            "ocr_image",
            "Extract text from an image using OCR. Params: file_path",
            dt.ocr_image,
        ))

    def _register_voice_tools(self):
        from tools.voice import VoiceTools
        vt = VoiceTools()

        self.register(Tool(
            "listen",
            "Listen to microphone and convert speech to text. Params: timeout (optional, default 5s)",
            vt.listen,
        ))
        self.register(Tool(
            "speak",
            "Convert text to speech and play it. Params: text",
            vt.speak,
        ))
        self.register(Tool(
            "save_speech",
            "Convert text to speech and save as MP3 file. Params: text, file_path (optional)",
            vt.save_speech,
        ))

    def _register_scheduler(self):
        from tools.scheduler import TaskScheduler
        self._scheduler = TaskScheduler()
        self._scheduler.load()

        self.register(Tool(
            "schedule_task",
            "Schedule a recurring task for the agent. Params: name, prompt, schedule (minutes or HH:MM), task_type (interval/daily)",
            self._scheduler.add_task,
        ))
        self.register(Tool(
            "list_scheduled_tasks",
            "List all scheduled tasks",
            lambda: "\n".join(f"{t['id']}: {t['name']} ({t['schedule']}) - enabled={t['enabled']}" for t in self._scheduler.list_tasks()) or "No tasks scheduled",
        ))
        self.register(Tool(
            "remove_scheduled_task",
            "Remove a scheduled task. Params: task_id",
            self._scheduler.remove_task,
        ))

    def _register_docker_tools(self):
        from tools.docker_sandbox import DockerSandbox
        ds = DockerSandbox()

        self.register(Tool(
            "docker_run",
            "Execute code in an isolated Docker container (secure). Params: code, language (python/bash/node, default python), timeout (default 30)",
            ds.run_code,
        ))
        self.register(Tool(
            "docker_images",
            "List available Docker images",
            ds.list_images,
        ))

    def _register_self_improve(self):
        from tools.self_improve import SelfImprover
        self._improver = SelfImprover()

        self.register(Tool(
            "self_analyze",
            "Analyze the agent's own codebase structure",
            self._improver.analyze_codebase,
        ))
        self.register(Tool(
            "self_review",
            "Run a full code review of the agent's own code and suggest improvements",
            self._improver.run_self_review,
        ))
        self.register(Tool(
            "suggest_improvements",
            "Suggest specific improvements for a code file. Params: file_path",
            self._improver.suggest_improvements,
        ))
        self.register(Tool(
            "apply_improvement",
            "Apply a code improvement to a file. Params: file_path, instructions",
            self._improver.apply_improvement,
        ))

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def format_for_prompt(self) -> str:
        if not self._tools:
            return ""
        lines = ["## Available Tools", ""]
        categories = {
            "Basic": ["datetime", "calculator", "run_code", "search_memory"],
            "Memory": ["recall", "remember"],
            "File Operations": ["read_file", "write_file", "edit_file", "glob", "grep", "list_dir", "file_info"],
            "Web": ["fetch_url", "search_web"],
            "Git": ["git_status", "git_diff", "git_log", "git_branch", "git_show"],
            "Code Analysis": ["scan_project", "review_code", "analyze_imports"],
            "Data Analysis": ["analyze_csv", "analyze_json", "analyze_text", "stats_summary"],
            "Documents & Images": ["read_pdf", "read_docx", "analyze_image", "ocr_image"],
            "Voice": ["listen", "speak", "save_speech"],
            "Multi-Agent": ["council", "delegate"],
            "Scheduling": ["schedule_task", "list_scheduled_tasks", "remove_scheduled_task"],
            "Docker Sandbox": ["docker_run", "docker_images"],
            "Self-Improvement": ["self_analyze", "self_review", "suggest_improvements", "apply_improvement"],
        }

        for cat, names in categories.items():
            tools_in_cat = [self._tools[n] for n in names if n in self._tools]
            if tools_in_cat:
                lines.append(f"### {cat}")
                for t in tools_in_cat:
                    lines.append(f"- `{t.name}`: {t.description[:120]}")
                lines.append("")

        lines.append("## How to use tools")
        lines.append('To use a tool, respond with: <tool name="tool_name">param1=value1|param2=value2</tool>')
        lines.append("Then wait for the result before continuing.")
        return "\n".join(lines)

    def parse_and_execute(self, text: str) -> list[dict]:
        results = []
        pattern = r'<tool name="([^"]+)">(.*?)</tool>'
        for match in re.finditer(pattern, text, re.DOTALL):
            name = match.group(1)
            params_str = match.group(2).strip()
            tool = self.get(name)
            if not tool:
                results.append({"tool": name, "result": f"Unknown tool: {name}"})
                continue

            kwargs = {}
            if params_str:
                for pair in params_str.split("|"):
                    pair = pair.strip()
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        kwargs[k.strip()] = v.strip()

            result = tool.run(**kwargs)
            results.append({"tool": name, "result": result})

        return results

    def contains_tool_call(self, text: str) -> bool:
        return '<tool name="' in text and "</tool>" in text
