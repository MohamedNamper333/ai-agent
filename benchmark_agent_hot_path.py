"""Round 7 benchmark: _is_simple_query + _parse_tool_calls with precompiled patterns.

Measures iterations/sec (not wall time) for micro-benchmark stability.
"""
import sys, time, json, re
from collections import namedtuple
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

sys.path.insert(0, r"E:\AI")

# ── Simulated data ────────────────────────────────────────────────────────────
ToolCall = namedtuple("ToolCall", "name arguments")
TaskStatus = MagicMock()  # placeholder; never inspected

# Simple queries (should be fast)
SIMPLE_QUERIES = [
    "read the file",
    "search the codebase",
    "list all files",
    "write a new file",
    "run the test suite",
    "find the bug",
    "commit changes",
    "push to remote",
    "create a branch",
    "summarize this",
]

# Complex queries (should also be fast but longer)
COMPLEX_QUERIES = [
    "Please analyze the entire codebase and identify all potential security vulnerabilities, "
    "then generate a comprehensive report with recommendations for each issue found.",
    "Refactor the authentication module to use JWT tokens instead of session-based auth, "
    "ensuring backward compatibility with existing API endpoints.",
]

# JSON payloads for parse_tool_calls
SIMPLE_TOOL_JSON = json.dumps({
    "tool_calls": [{"name": "read_file", "arguments": {"path": "/tmp/test.py"}}]
})
MULTI_TOOL_JSON = json.dumps({
    "tool_calls": [
        {"name": "read_file", "arguments": {"path": "/tmp/a.py"}},
        {"name": "write_file", "arguments": {"path": "/tmp/b.py"}, "content": "x=1"},
    ]
})
LEGACY_TOOL = '<tool name="run_code">command=echo hello</tool>'
NATIVE_TOOL = '{"name": "search_web", "arguments": {"query": "test"}}'
FENCED_TOOL = '```json\n' + SIMPLE_TOOL_JSON + '\n```'
TOOL_BLOCK = '<tool_call>{"name": "read_file", "arguments": {"path": "/tmp/x.py"}}</tool_call>'


def _is_simple_query(text: str) -> bool:
    words = text.split()
    if len(words) >= 15:
        return False
    text_lower = text.lower()
    return not any(kw in text_lower for kw in SIMPLE_QUERY_KEYWORDS)


# ── Module-level constants (copied from agent.py) ─────────────────────────────
SIMPLE_QUERY_KEYWORDS = frozenset({
    "read", "write", "edit", "search", "run", "execute", "find", "grep",
    "list", "create", "delete", "copy", "move", "compare", "batch",
    "git", "commit", "push", "pull", "docker", "schedule", "voice",
    "translate", "summarize", "استخرج", "ابحث", "شغل", "نفذ",
    "حمّل", "ارفع", "انشئ", "احذف", "عدّل", "اقرأ", "اكتب",
    "run_code", "read_file", "write_file", "edit_file", "search_web",
})
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


def _parse_tool_calls(text: str) -> list:
    calls = []
    for block in _RE_JSON_FENCE.findall(text):
        calls_from_block = _extract_tool_calls_from_json(block)
        if calls_from_block:
            return calls_from_block
    calls_from_json = _extract_tool_calls_from_json(text)
    if calls_from_json:
        return calls_from_json
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


def _extract_tool_calls_from_json(text: str) -> list:
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


# ── Benchmark harness ──────────────────────────────────────────────────────────
def bench(label: str, fn, data: list, rounds: int = 50_000) -> dict:
    """Run fn over data `rounds` times and report ops/sec + ns/op."""
    start = time.perf_counter()
    for _ in range(rounds):
        for item in data:
            fn(item)
    elapsed = time.perf_counter() - start
    total_ops = rounds * len(data)
    ns_per_op = (elapsed / total_ops) * 1e9
    ops_per_sec = total_ops / elapsed
    return {"label": label, "ops": total_ops, "elapsed_s": round(elapsed, 3),
            "ns_per_op": round(ns_per_op, 1), "ops_per_sec": round(ops_per_sec, 1)}


def main():
    print("=" * 70)
    print("Round 7 Benchmark: Precompiled patterns + frozenset")
    print("=" * 70)

    results = []
    results.append(bench("is_simple_query", _is_simple_query, SIMPLE_QUERIES, 200_000))
    results.append(bench("is_simple_query (complex)", _is_simple_query, COMPLEX_QUERIES, 200_000))
    results.append(bench("parse_tool_calls (json)", _parse_tool_calls, [SIMPLE_TOOL_JSON] * 5, 100_000))
    results.append(bench("parse_tool_calls (multi)", _parse_tool_calls, [MULTI_TOOL_JSON] * 5, 100_000))
    results.append(bench("parse_tool_calls (legacy)", _parse_tool_calls, [LEGACY_TOOL], 100_000))
    results.append(bench("parse_tool_calls (native)", _parse_tool_calls, [NATIVE_TOOL], 100_000))
    results.append(bench("parse_tool_calls (fenced)", _parse_tool_calls, [FENCED_TOOL], 100_000))
    results.append(bench("parse_tool_calls (block)", _parse_tool_calls, [TOOL_BLOCK], 100_000))

    print(f"\n{'Label':<32} {'ops':>12} {'ns/op':>10} {'ops/sec':>12}")
    print("-" * 70)
    for r in results:
        print(f"{r['label']:<32} {r['ops']:>12,} {r['ns_per_op']:>10.1f} {r['ops_per_sec']:>12,.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
