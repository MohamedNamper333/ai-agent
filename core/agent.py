from core.model import LLM
from core.memory import ConversationMemory
from core.tools import ToolRegistry
from core.context import ContextManager


class Agent:
    def __init__(
        self,
        model: LLM | None = None,
        memory: ConversationMemory | None = None,
        tools: ToolRegistry | None = None,
        context: ContextManager | None = None,
    ):
        self.model = model or LLM()
        self.memory = memory or ConversationMemory()
        self.tools = tools or ToolRegistry(agent=self)
        self.context = context or ContextManager()
        self.plugins = None

        self._load_plugins()
        self._start_scheduler()

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

    def _on_scheduled_task(self, name: str, prompt: str):
        print(f"\n[Scheduler] Running: {name}")
        result = self.chat(prompt, stream=False)
        print(f"[Scheduler] Done: {name}")
        return result

    def start_new_conversation(self, conversation_id: str = "") -> str:
        return self.memory.new_conversation(conversation_id)

    def chat(self, user_input: str, stream: bool = False):
        self.memory.add_message("user", user_input)
        self.memory.load()

        try:
            ltm = getattr(self.tools, '_ltm', None)
            if ltm:
                recall = ltm.get_context(user_input)
                if recall:
                    user_input = f"{recall}\n\nUser: {user_input}"
        except Exception:
            pass

        tool_desc = self.tools.format_for_prompt()
        history = self.memory.format_for_llm(self.context.system_prompt, include_system=False)

        if not stream:
            response = self._run_agent_loop(user_input, tool_desc, history)
            self.memory.add_message("assistant", response)
            self._auto_summarize(response, user_input)
            return response
        else:
            return self._stream_agent_loop(user_input, tool_desc, history)

    def _auto_summarize(self, response: str, user_input: str):
        try:
            msgs = self.memory.get_history()
            if len(msgs) > 20:
                ltm = getattr(self.tools, '_ltm', None)
                if ltm:
                    topic = user_input[:50].strip()
                    summary = f"User asked about: {topic[:80]}. Agent responded with {len(response)} chars."
                    ltm.add_summary(self.memory.current_id, summary, [topic])
        except Exception:
            pass

    def _build_system_prompt(self, user_input: str, tool_desc: str, history: str,
                              tool_results: list[dict] | None = None) -> str:
        sp = self.context.system_prompt
        parts = []
        parts.append(f"<|system|>\n{sp}\n")
        parts.append(f"<|system|>\nYou have access to the following tools. Use them when the user's request requires it.\n{tool_desc}\n")
        parts.append(
            "<|system|>\nGuidelines:\n"
            "- When you need information or need to perform an action, use a tool.\n"
            "- After getting tool results, use them to respond to the user.\n"
            "- You can use multiple tools in sequence if needed.\n"
            "- Always respond with helpful information after using tools.\n"
        )
        if history:
            parts.append(history)
        parts.append(f"<|user|>\n{user_input}\n")
        if tool_results:
            ctx = "\n".join(f"Tool '{r['tool']}' returned:\n{r['result']}\n" for r in tool_results)
            parts.append(f"Tool results:\n{ctx}\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)

    def _run_agent_loop(self, user_input: str, tool_desc: str, history: str) -> str:
        prompt = self._build_system_prompt(user_input, tool_desc, history)
        response = self.model.generate(prompt)

        max_loops = 8
        loop_count = 0
        while self.tools.contains_tool_call(response) and loop_count < max_loops:
            loop_count += 1
            tool_results = self.tools.parse_and_execute(response)
            for tr in tool_results:
                self.memory.add_message("system", f"Tool '{tr['tool']}': {tr['result'][:300]}")
            prompt = self._build_system_prompt(
                user_input, tool_desc, history, tool_results
            )
            response = self.model.generate(prompt)
        return response

    def _stream_agent_loop(self, user_input: str, tool_desc: str, history: str):
        prompt = self._build_system_prompt(user_input, tool_desc, history)
        full_response = ""
        for chunk in self.model.generate(prompt, stream=True):
            full_response += chunk
            yield chunk

        max_loops = 5
        loop_count = 0
        while self.tools.contains_tool_call(full_response) and loop_count < max_loops:
            loop_count += 1
            tool_results = self.tools.parse_and_execute(full_response)
            for tr in tool_results:
                self.memory.add_message("system", f"Tool '{tr['tool']}': {tr['result'][:300]}")
            prompt = self._build_system_prompt(
                user_input, tool_desc, history, tool_results
            )
            full_response = ""
            for chunk in self.model.generate(prompt, stream=True):
                full_response += chunk
                yield chunk
        self.memory.add_message("assistant", full_response)

    def get_history(self) -> list[dict]:
        return self.memory.get_trimmed_history()
