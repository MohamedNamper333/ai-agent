"""core/tools_base.py — Tool and ToolResult data classes.

Extracted from core/tools.py (was 850 lines).
All existing imports still work:
    from core.tools import Tool, ToolResult   ← re-exported from tools.py
    from core.tools_base import Tool, ToolResult  ← direct import
"""
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class ToolResult:
    """Result returned by any tool execution."""

    tool_name: str
    success: bool
    result: str
    error: str = ""
    execution_time: float = 0.0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.strftime("%H:%M:%S")

    def __bool__(self) -> bool:
        return self.success


class Tool:
    """Wraps a callable function as an agent-usable tool."""

    def __init__(
        self,
        name: str,
        description: str,
        func: Callable,
        timeout: int = 30,
        requires_auth: bool = False,
        category: str = "general",
    ):
        self.name = name
        self.description = description
        self.func = func
        self.timeout = timeout
        self.requires_auth = requires_auth
        self.category = category

        # Stats
        self._call_count: int = 0
        self._error_count: int = 0
        self._total_time: float = 0.0

    def run(self, **kwargs: Any) -> ToolResult:
        """Execute the tool and return a ToolResult."""
        start = time.time()
        self._call_count += 1
        try:
            result = self.func(**kwargs)
            elapsed = time.time() - start
            self._total_time += elapsed
            return ToolResult(
                tool_name=self.name,
                success=True,
                result=str(result),
                execution_time=elapsed,
            )
        except Exception as exc:
            elapsed = time.time() - start
            self._total_time += elapsed
            self._error_count += 1
            return ToolResult(
                tool_name=self.name,
                success=False,
                result="",
                error=str(exc),
                execution_time=elapsed,
            )

    def get_stats(self) -> dict:
        """Return hit rate, miss count, eviction count, and current size."""
        avg = self._total_time / max(self._call_count, 1)
        err_rate = self._error_count / max(self._call_count, 1) * 100
        return {
            "name": self.name,
            "calls": self._call_count,
            "errors": self._error_count,
            "error_rate": f"{err_rate:.1f}%",
            "avg_time": f"{avg:.3f}s",
            "total_time": f"{self._total_time:.3f}s",
        }

    def __repr__(self) -> str:
        return f"Tool(name={self.name!r}, category={self.category!r})"
