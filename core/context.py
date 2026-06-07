import json
from typing import Optional

import config


class ContextManager:
    def __init__(self, max_tokens: int = 0):
        self.max_tokens = max_tokens or config.N_CTX
        self.system_prompt = config.SYSTEM_PROMPT
        self.tool_outputs: list[dict] = []
        self._context_cache: dict[str, str] = {}

    def build_prompt(
        self,
        user_input: str,
        history: str = "",
        tool_descriptions: str = "",
        system_prompt: Optional[str] = None,
    ) -> str:
        sp = system_prompt if system_prompt is not None else self.system_prompt

        parts = []
        parts.append(f"<|system|>\n{sp}\n")

        if tool_descriptions:
            tool_section = (
                "\nYou have access to the following tools. Use them when needed / لديك الأدوات التالية. استخدمها عند الحاجة:\n"
                f"{tool_descriptions}\n"
            )
            parts.append(f"<|system|>\n{tool_section}\n")

        if history:
            parts.append(history)

        parts.append(f"<|user|>\n{user_input}\n")
        parts.append("<|assistant|>\n")

        return "".join(parts)

    def build_with_tool_results(
        self,
        user_input: str,
        tool_results: list[dict],
        history: str = "",
        tool_descriptions: str = "",
    ) -> str:
        sp = self.system_prompt

        parts = []
        parts.append(f"<|system|>\n{sp}\n")

        if tool_descriptions:
            parts.append(f"<|system|>\n{tool_descriptions}\n")

        if history:
            parts.append(history)

        tool_context_parts = []
        for r in tool_results:
            status = "OK" if r.get("success", True) else "ERROR"
            tool_context_parts.append(
                f"[{status}] Tool '{r['tool']}' returned:\n{r['result'][:2000]}"
            )
        tool_context = "\n\n".join(tool_context_parts)

        parts.append(f"<|user|>\n{user_input}\n\nTool results:\n{tool_context}\n")
        parts.append("<|assistant|>\n")

        return "".join(parts)

    def build_plan_prompt(
        self,
        user_input: str,
        tool_descriptions: str = "",
        history: str = "",
    ) -> str:
        sp = self.system_prompt

        parts = []
        parts.append(
            f"<|system|>\n{sp}\n"
            "You are a planning agent. Analyze the request and create an execution plan.\n"
        )

        if tool_descriptions:
            parts.append(f"<|system|>\nAvailable tools:\n{tool_descriptions}\n")

        if history:
            parts.append(history)

        parts.append(
            "<|system|>\n"
            'Respond with JSON: {"goal": "...", "steps": [{"description": "...", '
            '"tool_calls": [{"name": "tool_name", "arguments": {"param": "value"}}]}]}\n'
            "If no tools needed: {\"goal\": \"...\", \"steps\": []}\n"
        )

        parts.append(f"<|user|>\n{user_input}\n")
        parts.append("<|assistant|>\n")

        return "".join(parts)

    def count_tokens(self, text: str, model) -> int:
        try:
            return model.count_tokens(text)
        except Exception:
            return len(text) // 4

    def get_stats(self) -> dict:
        return {
            "max_tokens": self.max_tokens,
            "system_prompt_length": len(self.system_prompt),
            "tool_outputs_count": len(self.tool_outputs),
            "context_cache_size": len(self._context_cache),
        }
