import json
import re
import time
import hashlib
from typing import Optional, Generator
from dataclasses import dataclass, field
from enum import Enum

import config

# ---------------------------------------------------------------------------
# Pre-compiled regex patterns for parse_tool_calls / extract_json_from_text
# ---------------------------------------------------------------------------
_RE_JSON_FENCE = re.compile(r'```(?:json)?\s*\n?(.*?)\n?```', re.DOTALL)
_RE_TOOL_CALL_NATIVE = re.compile(
    r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{.*?\})\s*\}', re.DOTALL
)
_RE_TOOL_LEGACY = re.compile(
    r'<tool[^>]*name="([^"]+)"[^>]*>(.*?)</tool>', re.DOTALL
)
_RE_TOOL_BLOCK = re.compile(
    r'<tool_call>\s*\{.*?["\']name["\']\s*:\s*["\']([^"\']+)["\'].*?\}\s*</tool_call>',
    re.DOTALL,
)
_RE_TOOL_BLOCK_JSON = re.compile(r'\{.*\}', re.DOTALL)
_RE_JSON_EXTRACT = re.compile(r'\{[^{}]*"tool_calls"\s*:\s*\[.*?\]\s*\}', re.DOTALL)
_RE_JSON_SEARCH = re.compile(r'\{.*\}', re.DOTALL)
_RE_ANY_NAME = re.compile(r'"name"\s*:\s*"([^"]+)"')

# ---------------------------------------------------------------------------
# Pre-built keyword set for _is_simple_query
# ---------------------------------------------------------------------------
_SIMPLE_QUERY_KEYWORDS = frozenset({
    "read", "write", "edit", "search", "run", "execute", "find", "grep",
    "list", "create", "delete", "copy", "move", "compare", "batch",
    "git", "commit", "push", "pull", "docker", "schedule", "voice",
    "translate", "summarize", "استخرج", "ابحث", "شغل", "نفذ",
    "حمّل", "ارفع", "انشئ", "احذف", "عدّل", "اقرأ", "اكتب",
    "run_code", "read_file", "write_file", "edit_file", "search_web",
})

from core.llm import LLMRouter
from core.telemetry import Telemetry
from core.reasoning import CoTEngine, ReasoningChain
from core.memory import ConversationMemory
from core.tools import ToolRegistry, Tool
from core.context import ContextManager
from core.cache import get_cache_manager, make_cache_key
from rag.retriever import Retriever


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str = ""
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    error: str = ""
    attempts: int = 0
    max_attempts: int = 3
    timestamp: str = ""

    def __post_init__(self):
        if not self.id:
            unique_str = f"{self.name}_{time.time()}"
            self.id = f"call_{hashlib.md5(unique_str.encode()).hexdigest()[:8]}"
        if not self.timestamp:
            self.timestamp = time.strftime("%H:%M:%S")


@dataclass
class PlanStep:
    description: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: str = ""
    reflection: str = ""


@dataclass
class ExecutionPlan:
    goal: str
    steps: list[PlanStep] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    final_result: str = ""
    total_tool_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0


class Agent:
    MAX_TOOL_LOOPS = 10
    MAX_RETRIES = 3
    RETRY_BACKOFF_BASE = 2
    CONTEXT_RESERVED_TOKENS = 1500

    def __init__(
        self,
        model: LLMRouter | None = None,
        memory: ConversationMemory | None = None,
        tools: ToolRegistry | None = None,
        context: ContextManager | None = None,
        telemetry: Telemetry | None = None,
        cot_engine: CoTEngine | None = None,
    ):
        self.model = model or LLMRouter()
        self.memory = memory or ConversationMemory()
        self.tools = tools or ToolRegistry(agent=self)
        self.context = context or ContextManager()
        self.telemetry = telemetry or Telemetry()
        self.cot = cot_engine or CoTEngine(self.model)
        self.plugins = None
        self._execution_history: list[ExecutionPlan] = []
        self._cache = get_cache_manager().get_cache("agent_responses", max_size=500, ttl=config.CACHE_TTL)
        self._retriever: Optional[Retriever] = None
        self._fast_mode = config.FAST_MODE

        # 4 Pillars
        self._init_pillars()

        self._load_plugins()
        self._start_scheduler()
        self._init_rag()

    def _init_pillars(self) -> None:
        try:
            from core.reasoning.deductive_engine import DeductiveEngine
            self.deductive = DeductiveEngine(model=self.model)
        except Exception:
            self.deductive = None

        try:
            from core.memory.neural_memory import get_neural_memory
            self.neural_memory = get_neural_memory()
        except Exception:
            self.neural_memory = None

        try:
            from core.memory.obsidian_bridge import get_obsidian_bridge
            self.obsidian = get_obsidian_bridge()
        except Exception:
            self.obsidian = None

        try:
            from core.learning_engine import get_learning_engine
            self.learning = get_learning_engine()
        except Exception:
            self.learning = None

    def think(self, query: str, level: str = "moderate") -> ReasoningChain:
        """Delegate chain-of-thought reasoning to the CoT engine.

        Args:
            query: The user query or task to reason about.
            level: One of 'simple', 'moderate', 'deep'.

        Returns:
            ReasoningChain with parsed steps and final answer.
        """
        with self.telemetry.track("think", level=level, query_len=len(query)) as evt:
            chain = self.cot.think(query, level=level)
            evt.data["steps"] = len(chain.steps)
            evt.data["avg_confidence"] = chain.avg_confidence
            return chain

    def _load_plugins(self):
        try:
            from plugins import PluginRegistry
            self.plugins = PluginRegistry()
            self.plugins.discover_and_load(self)
            for plugin in self.plugins.plugins:
                for tool_def in plugin.get_tools():
                    from core.tools import Tool
                    self.tools.register(Tool(
                        tool_def["name"],
                        tool_def["description"],
                        tool_def["func"],
                    ))
        except Exception as e:
            print(f"[agent] Plugin load: {e}")

    def _start_scheduler(self):
        try:
            scheduler = getattr(self.tools, '_scheduler', None)
            if scheduler:
                scheduler.set_callback(self._on_scheduled_task)
                scheduler.start()
        except Exception:
            pass

    def _init_rag(self):
        if config.RAG_ENABLED:
            try:
                self._retriever = Retriever()
                self._retriever.load_or_init()
            except Exception:
                self._retriever = None

    def _is_simple_query(self, text: str) -> bool:
        words = text.split()
        if len(words) >= 15:
            return False
        text_lower = text.lower()
        return not any(kw in text_lower for kw in _SIMPLE_QUERY_KEYWORDS)

    def _on_scheduled_task(self, name: str, prompt: str):
        print(f"\n[Scheduler] Running: {name}")
        result = self.chat(prompt, stream=False)
        print(f"[Scheduler] Done: {name}")
        return result

    def start_new_conversation(self, conversation_id: str = "") -> str:
        """Start a fresh conversation and return its ID."""
        return self.memory.new_conversation(conversation_id)

    def chat(self, user_input: str, stream: bool = False):
        """Process a user message and return the agent response (stream or string)."""
        self.memory.add_message("user", user_input)

        # RAG: retrieve knowledge
        rag_context = ""
        if config.RAG_ENABLED and self._retriever:
            try:
                rag_ctx = self._retriever.query_text(user_input, top_k=2)
                if rag_ctx:
                    rag_context = f"\n[Retrieved knowledge]\n{rag_ctx}\n"
            except Exception:
                pass

        # Cache: check for cached response (non-stream only)
        cache_key = ""
        if not stream and config.CACHE_TTL > 0:
            tool_names = sorted(t.name for t in self.tools.list_tools())
            cache_key = make_cache_key(user_input, "|".join(tool_names), rag_context[:200])
            cached = self._cache.get(cache_key)
            if cached is not None:
                self.memory.add_message("assistant", cached)
                return cached


        # Pillar 2: Neural Memory recall
        neural_ctx = self._recall_neural_memory(user_input)
        if neural_ctx:
            user_input = f"{neural_ctx}\n\nUser: {user_input}"

        ltm_recall = self._recall_ltm(user_input)
        if ltm_recall:
            user_input = f"{ltm_recall}\n\nUser: {user_input}"

        # Enrich with RAG
        if rag_context:
            user_input = f"{rag_context}\n\nUser question: {user_input}"

        tool_desc = self.tools.format_for_prompt()
        history = self.memory.format_for_llm(self.context.system_prompt, include_system=False)

        # Fast mode: skip planning for simple queries
        use_fast = False
        if self._fast_mode == "on" or (self._fast_mode == "auto" and self._is_simple_query(user_input)):
            use_fast = True

        if not stream:
            if use_fast:
                prompt = self._build_system_prompt(user_input, tool_desc, history)
                response = self.model.generate(prompt)
            else:
                response = self._execute_with_plan(user_input, tool_desc, history)

            self.memory.add_message("assistant", response)
            self._auto_summarize(response, user_input)
            # Pillar 3: Learning Engine — capture interaction
            self._capture_interaction(user_input, response)
            # Cache the response
            if cache_key:
                self._cache.set(cache_key, response)
            return response
        else:
            if use_fast:
                prompt = self._build_system_prompt(user_input, tool_desc, history)
                return self.model.generate(prompt, stream=True)
            return self._stream_agent_loop(user_input, tool_desc, history)

    async def achat(self, user_input: str, stream: bool = False):
        """Async wrapper around chat() for use in async contexts."""
        import asyncio
        if stream:
            return self.chat(user_input, stream=True)
        return await asyncio.to_thread(self.chat, user_input, False)

    def _recall_ltm(self, user_input: str) -> str:
        try:
            ltm = getattr(self.tools, '_ltm', None)
            if ltm:
                return ltm.get_context(user_input)
        except Exception:
            pass
        return ""

    def _recall_neural_memory(self, user_input: str) -> str:
        """Retrieve relevant memories from Neural Memory (Pillar 2)."""
        try:
            if not self.neural_memory:
                return ""
            from core.memory.neural_memory import MemoryQuery
            results = self.neural_memory.recall(MemoryQuery(
                query_text=user_input, top_k=3, min_importance=0.3
            ))
            if not results:
                return ""
            parts = ["[Memory context]"]
            for r in results:
                if r.relevance_score > 0.3:
                    parts.append(f"- {r.node.content[:120]}")
                    if r.node.reasoning:
                        parts.append(f"  (reason: {r.node.reasoning[:60]})")
            return "\n".join(parts) if len(parts) > 1 else ""
        except Exception:
            return ""

    def _capture_interaction(self, user_input: str, response: str) -> None:
        """Capture interaction for learning (Pillar 3)."""
        try:
            if self.learning:
                self.learning.capture(
                    user_input=user_input,
                    assistant_response=response,
                    tools_used=[t.name for t in self.tools.list_tools()[:5]],
                    success=bool(response and len(response) > 10),
                )
            if self.neural_memory and len(response) > 50:
                self.neural_memory.remember(
                    content=f"Q: {user_input[:100]} A: {response[:200]}",
                    node_type="observation",
                    importance=0.4,
                )
            if self.obsidian:
                self.obsidian.log_daily(
                    summary=f"Q: {user_input[:80]}",
                    decisions_made=0,
                    tools_used=[],
                )
        except Exception:
            pass

    def think_deep(self, problem: str, context: str = "") -> str:
        """Full deductive reasoning on a complex problem (Pillar 1)."""
        try:
            if self.deductive:
                result = self.deductive.think(problem, context)
                # Store decision in neural memory
                if self.neural_memory:
                    self.neural_memory.remember_decision(
                        decision=result.chosen_plan.description,
                        reasoning=result.reasoning,
                        factors=result.chosen_plan.steps[:3],
                        context=problem[:200],
                        importance=result.chosen_plan.composite_score,
                    )
                # Store in Obsidian
                if self.obsidian:
                    self.obsidian.write_decision(
                        title=problem[:60],
                        decision=result.chosen_plan.description,
                        reasoning=result.reasoning[:300],
                        factors=result.chosen_plan.steps[:5],
                        alternatives=[result.alternative_considered[:200]],
                    )
                return result.to_report()
        except Exception as e:
            pass
        return self.think(problem, level="deep").result

    def get_pillars_status(self) -> dict:
        """Return health status of all 4 pillars."""
        return {
            "deductive_engine": self.deductive is not None,
            "neural_memory": self.neural_memory is not None,
            "obsidian_bridge": self.obsidian is not None,
            "learning_engine": self.learning is not None,
            "neural_memory_stats": self.neural_memory.get_stats() if self.neural_memory else {},
            "learning_stats": self.learning.get_stats() if self.learning else {},
            "obsidian_stats": self.obsidian.get_stats() if self.obsidian else {},
        }



    def _auto_summarize(self, response: str, user_input: str):
        try:
            msgs = self.memory.get_history()
            if len(msgs) > 15:
                ltm = getattr(self.tools, '_ltm', None)
                if ltm:
                    # Use a small portion of the conversation for context-aware summarization
                    context_for_summary = f"User: {user_input}\nAgent: {response[:300]}"
                    topic = user_input[:50].strip()
                    summary = f"Summary: {response[:150].strip()}... | Context: {context_for_summary[:100]}"
                    ltm.add_summary(self.memory.current_id, summary, [topic])
        except Exception:
            pass

    def _get_available_tools(self) -> list[dict]:
        tools_info = []
        for tool in self.tools.list_tools():
            tools_info.append({
                "name": tool.name,
                "description": tool.description,
            })
        return tools_info

    def _build_system_prompt(self, user_input: str, tool_desc: str, history: str,
                               tool_results: list[dict] | None = None,
                               plan_context: str = "") -> str:
        if tool_results:
            return self.context.build_with_tool_results(
                user_input, tool_results, history, tool_desc
            )
        return self.context.build_prompt(user_input, history, tool_desc)

    def _parse_tool_calls(self, text: str) -> list[ToolCall]:
        calls = []

        # 1. Extract JSON from code fences (models often wrap in ```json ... ```)
        for block in _RE_JSON_FENCE.findall(text):
            calls_from_block = self._extract_tool_calls_from_json(block)
            if calls_from_block:
                return calls_from_block

        # 2. Direct JSON in response
        calls_from_json = self._extract_tool_calls_from_json(text)
        if calls_from_json:
            return calls_from_json

        # 3. Native function calling format: {"name": "...", "arguments": {...}}
        for match in _RE_TOOL_CALL_NATIVE.finditer(text):
            try:
                name = match.group(1)
                args_str = match.group(2).strip()
                args = json.loads(args_str)
                calls.append(ToolCall(name=name, arguments=args))
            except (json.JSONDecodeError, KeyError):
                continue

        if calls:
            return calls

        # 4. Legacy <tool> format
        for match in _RE_TOOL_LEGACY.finditer(text):
            name = match.group(1)
            params_str = match.group(2).strip()
            kwargs = {}
            if params_str:
                for pair in params_str.split("|"):
                    pair = pair.strip()
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        kwargs[k.strip()] = v.strip()
            calls.append(ToolCall(name=name, arguments=kwargs))

        # 5. <tool_call> blocks (used by some fine-tuned models)
        if not calls:
            for match in _RE_TOOL_BLOCK.finditer(text):
                try:
                    name = match.group(1)
                    json_part = _RE_TOOL_BLOCK_JSON.search(match.group())
                    if json_part:
                        data = json.loads(json_part.group())
                        args = data.get("arguments", {})
                        calls.append(ToolCall(name=name, arguments=args))
                except (json.JSONDecodeError, KeyError):
                    continue

        return calls

    def _extract_tool_calls_from_json(self, text: str) -> list[ToolCall]:
        calls = []
        try:
            data = json.loads(text)
            if "tool_calls" in data and isinstance(data["tool_calls"], list):
                for tc in data["tool_calls"]:
                    if isinstance(tc, dict):
                        name = tc.get("name") or tc.get("function", {}).get("name", "")
                        args = tc.get("arguments") or tc.get("function", {}).get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                args = {}
                        if name:
                            calls.append(ToolCall(name=name, arguments=args))
                return calls
            if "name" in data and data.get("name"):
                name = data["name"]
                args = data.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                calls.append(ToolCall(name=name, arguments=args))
                return calls
        except (json.JSONDecodeError, KeyError):
            pass

        for match in _RE_JSON_EXTRACT.finditer(text):
            try:
                data = json.loads(match.group())
                if "tool_calls" in data and isinstance(data["tool_calls"], list):
                    for tc in data["tool_calls"]:
                        if isinstance(tc, dict):
                            name = tc.get("name", "")
                            args = tc.get("arguments", {})
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except json.JSONDecodeError:
                                    args = {}
                            if name:
                                calls.append(ToolCall(name=name, arguments=args))
                    if calls:
                        return calls
            except (json.JSONDecodeError, KeyError):
                continue

        return calls

    def _execute_tool_call(self, call: ToolCall) -> ToolCall:
        tool = self.tools.get(call.name)
        if not tool:
            call.status = TaskStatus.FAILED
            call.error = f"Unknown tool: {call.name}"
            return call
        call.attempts += 1
        try:
            result = tool.run(**call.arguments)
            if hasattr(result, 'success'):
                call.result = result.result[:5000] if result.success else ""
                if not result.success:
                    call.error = result.error
                    call.status = TaskStatus.FAILED
                    return call
            else:
                call.result = str(result)[:5000]
            call.status = TaskStatus.COMPLETED
        except Exception as e:
            call.error = str(e)
            call.status = TaskStatus.FAILED
        return call

    def _run_tool_calls(self, calls: list[ToolCall]) -> list[dict]:
        results = []
        for call in calls:
            self._execute_tool_call_with_retry(call)
            results.append({
                "tool": call.name,
                "result": call.result if call.status == TaskStatus.COMPLETED else f"ERROR: {call.error}",
                "error": call.error,
                "success": call.status == TaskStatus.COMPLETED
            })
        return results

    def _execute_tool_call_with_retry(self, call: ToolCall) -> ToolCall:
        for attempt in range(call.max_attempts):
            call.attempts = attempt + 1
            self._execute_tool_call(call)
            if call.status == TaskStatus.COMPLETED or attempt >= call.max_attempts - 1:
                break
            time.sleep(min(self.RETRY_BACKOFF_BASE ** attempt, 5))
        return call

    def _execute_with_plan(self, user_input: str, tool_desc: str, history: str) -> str:
        plan = self._create_plan(user_input, tool_desc, history)
        if not plan.steps:
            prompt = self._build_system_prompt(user_input, tool_desc, history)
            return self.model.generate(prompt)

        for step in plan.steps:
            step.status = TaskStatus.IN_PROGRESS
            if step.tool_calls:
                results = self._run_tool_calls(step.tool_calls)
                plan.total_tool_calls += len(results)
                plan.successful_calls += sum(1 for r in results if r["success"])
                plan.failed_calls += sum(1 for r in results if not r["success"])
            step.result = self._synthesize_step_result(step, user_input, tool_desc, history)
            step.status = TaskStatus.COMPLETED

        plan.status = TaskStatus.COMPLETED
        plan.final_result = self._synthesize_plan_result(plan, user_input, tool_desc, history)
        self._execution_history.append(plan)
        return plan.final_result

    def _create_plan(self, user_input: str, tool_desc: str, history: str) -> ExecutionPlan:
        plan_prompt = (
            f"<|system|>\nYou are a planning agent. Analyze the user's request and create an execution plan.\n\n"
            f"## Available Tools\n{tool_desc}\n\n"
            f"## Planning Instructions\n"
            f"1. Understand the user's goal\n"
            f"2. Break it into sequential steps\n"
            f"3. For each step, decide which tool (if any) is needed\n"
            f"4. If multiple tools can run in parallel, list them in the same step\n"
            f"5. Set realistic expectations\n\n"
            f"## Output Format\n"
            f'{{\n  "goal": "Clear description of the overall goal",\n'
            f'  "steps": [\n'
            f'    {{\n'
            f'      "description": "What this step accomplishes",\n'
            f'      "tool_calls": [\n'
            f'        {{"name": "tool_name", "arguments": {{"param": "value"}}}}\n'
            f'      ]\n'
            f'    }}\n'
            f'  ]\n'
            f'}}\n\n'
            f"If no tools are needed, return: {{\"goal\": \"...\", \"steps\": []}}\n"
            f"<|user|>\n{user_input}\n<|assistant|>\n"
        )

        response = self.model.generate(plan_prompt, max_tokens=1000)

        plan = ExecutionPlan(goal=user_input)

        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                plan.goal = data.get("goal", user_input)

                for step_data in data.get("steps", []):
                    tool_calls = []
                    for tc in step_data.get("tool_calls", []):
                        tool_calls.append(ToolCall(
                            name=tc.get("name", ""),
                            arguments=tc.get("arguments", {}),
                        ))
                    step = PlanStep(
                        description=step_data.get("description", ""),
                        tool_calls=tool_calls,
                    )
                    plan.steps.append(step)
        except (json.JSONDecodeError, KeyError):
            pass

        if not plan.steps:
            all_calls = self._parse_tool_calls(response)
            if all_calls:
                plan.steps.append(PlanStep(
                    description="Execute tools",
                    tool_calls=all_calls,
                ))

        return plan

    def _synthesize_step_result(self, step: PlanStep, user_input: str, tool_desc: str, history: str) -> str:
        completed_calls = [c for c in step.tool_calls if c.status == TaskStatus.COMPLETED]
        failed_calls = [c for c in step.tool_calls if c.status == TaskStatus.FAILED]

        if not completed_calls and not failed_calls:
            return f"Step '{step.description}': No tools were executed."

        results_parts = []
        for call in completed_calls:
            results_parts.append(f"Tool '{call.name}' result:\n{call.result[:2000]}")
        for call in failed_calls:
            results_parts.append(f"Tool '{call.name}' FAILED: {call.error}")

        results_text = "\n\n".join(results_parts)

        synth_prompt = (
            f"<|system|>\nYou are summarizing the results of a step in an execution plan.\n"
            f"Step: {step.description}\n\nTool Results:\n{results_text}\n\n"
            f"Provide a clear summary of what was accomplished.\n"
            f"<|user|>\nSummarize these results.\n<|assistant|>\n"
        )

        return self.model.generate(synth_prompt, max_tokens=500)

    def _synthesize_plan_result(self, plan: ExecutionPlan, user_input: str, tool_desc: str, history: str) -> str:
        step_summaries = []
        for i, step in enumerate(plan.steps, 1):
            status = "OK" if step.status == TaskStatus.COMPLETED else "FAILED"
            step_summaries.append(f"[{status}] Step {i}: {step.description}\nResult: {step.result[:1000]}")

        steps_text = "\n\n".join(step_summaries)

        stats = (
            f"Plan Statistics: {plan.total_tool_calls} tool calls, "
            f"{plan.successful_calls} successful, {plan.failed_calls} failed"
        )

        synth_prompt = (
            f"<|system|>\nYou are synthesizing the final result of an execution plan.\n"
            f"Goal: {plan.goal}\n\n{stats}\n\nSteps:\n{steps_text}\n\n"
            f"Provide a comprehensive final response to the user's request.\n"
            f"<|user|>\nSynthesize the final result.\n<|assistant|>\n"
        )

        return self.model.generate(synth_prompt, max_tokens=1500)

    def _stream_agent_loop(self, user_input: str, tool_desc: str, history: str):
        prompt = self._build_system_prompt(user_input, tool_desc, history)
        full_response = ""
        for chunk in self.model.generate(prompt, stream=True):
            full_response += chunk
            yield chunk

        for loop_count in range(self.MAX_TOOL_LOOPS):
            calls = self._parse_tool_calls(full_response)
            if not calls:
                break
            tool_results = self._run_tool_calls(calls)
            for tr in tool_results:
                self.memory.add_message("system", f"Tool '{tr['tool']}': {tr['result'][:300]}")
            prompt = self._build_system_prompt(user_input, tool_desc, history, tool_results)
            full_response = ""
            for chunk in self.model.generate(prompt, stream=True):
                full_response += chunk
                yield chunk

        self.memory.add_message("assistant", full_response)

    def get_history(self) -> list[dict]:
        """Return the last N execution history entries."""
        return self.memory.get_trimmed_history()

    def get_cache_stats(self) -> dict:
        """Get cache stats."""
        return self._cache.get_stats()

    def get_execution_history(self) -> list[dict]:
        """Get execution history."""
        history = []
        for plan in self._execution_history:
            history.append({
                "goal": plan.goal,
                "status": plan.status.value,
                "total_calls": plan.total_tool_calls,
                "successful": plan.successful_calls,
                "failed": plan.failed_calls,
                "steps": len(plan.steps),
            })
        return history
