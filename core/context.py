import json
from typing import Optional

import config


class ContextManager:
    def __init__(self, max_tokens: int = 0):
        self.max_tokens = max_tokens or config.N_CTX
        self.system_prompt = config.SYSTEM_PROMPT
        self.tool_outputs: list[dict] = []

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
                "\nYou have access to the following tools. Use them when needed:\n"
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

        tool_context = "\n".join(
            f"Tool '{r['tool']}' returned:\n{r['result']}\n"
            for r in tool_results
        )
        parts.append(f"<|user|>\n{user_input}\n\nTool results:\n{tool_context}\n")
        parts.append("<|assistant|>\n")

        return "".join(parts)

    def count_tokens(self, text: str, model) -> int:
        try:
            return model.count_tokens(text)
        except Exception:
            return len(text) // 4
