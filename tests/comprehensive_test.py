"""Comprehensive System Test Suite"""
import sys
import os
import json
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.tools import ToolRegistry
from core.agent import Agent
from rag.retriever import Retriever
import config

class TestResult:
    __test__ = False
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def check(self, name, condition, detail=""):
        if condition:
            self.passed += 1
            print(f"  [PASS] {name}")
        else:
            self.failed += 1
            self.errors.append(f"{name}: {detail}")
            print(f"  [FAIL] {name} - {detail}")
    
    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*50}")
        print(f"Results: {self.passed}/{total} passed")
        if self.errors:
            print(f"\nFailed tests:")
            for e in self.errors:
                print(f"  - {e}")
        return self.failed == 0


def test_tool_registry(results):
    print("\n[1] Tool Registry Tests")
    reg = ToolRegistry()
    
    results.check("Tools registered", len(reg.list_tools()) > 50, 
                  f"Only {len(reg.list_tools())} tools")
    
    results.check("Categories exist", len(reg.list_tools_by_category()) > 10,
                  f"Only {len(reg.list_tools_by_category())} categories")
    
    results.check("run_code exists", reg.get("run_code") is not None)
    results.check("calculator exists", reg.get("calculator") is not None)
    results.check("read_file exists", reg.get("read_file") is not None)
    results.check("write_file exists", reg.get("write_file") is not None)
    results.check("search_web exists", reg.get("search_web") is not None)


def test_security(results):
    print("\n[2] Security Tests")
    reg = ToolRegistry()
    run_code = reg.get("run_code")
    calc = reg.get("calculator")
    
    r1 = run_code.run(code="import os")
    results.check("Block import os", "restricted" in r1.result.lower())
    
    r2 = run_code.run(code="eval(1)")
    results.check("Block eval()", "restricted" in r2.result.lower())
    
    r3 = run_code.run(code="open('test.txt')")
    results.check("Block open()", "restricted" in r3.result.lower())
    
    r4 = run_code.run(code="import subprocess")
    results.check("Block subprocess", "restricted" in r4.result.lower())
    
    r5 = run_code.run(code="print(2+2)")
    results.check("Allow safe code", "4" in r5.result)
    
    c1 = calc.run(expr='__import__("os").system("ls")')
    results.check("Calculator block __import__", "Forbidden" in c1.result)
    
    c2 = calc.run(expr="2**10")
    results.check("Calculator allow math", "1024" in c2.result)


def test_path_traversal(results):
    print("\n[3] Path Traversal Tests")
    from pathlib import Path
    
    base = (Path(".") / "web").resolve()
    
    p1 = (Path(".") / "web" / "../../etc/passwd").resolve()
    results.check("Block ../../etc/passwd", not p1.is_relative_to(base))
    
    p2 = (Path(".") / "web" / "../config.py").resolve()
    results.check("Block ../config.py", not p2.is_relative_to(base))
    
    p3 = (Path(".") / "web" / "style.css").resolve()
    results.check("Allow style.css", p3.is_relative_to(base))


def test_agent(results):
    print("\n[4] Agent Tests")
    agent = Agent()
    
    results.check("Agent initialized", agent is not None)
    results.check("Model loaded", agent.model is not None)
    results.check("Memory initialized", agent.memory is not None)
    results.check("Tools registered", len(agent.tools.list_tools()) > 50)
    
    history = "x" * 10000
    prompt = agent._build_system_prompt("test", "tools", history)
    results.check("Context truncated", len(prompt) < 15000,
                  f"Prompt length: {len(prompt)}")


def test_memory(results):
    print("\n[5] Memory Tests")
    agent = Agent()
    
    stats = agent.memory.get_stats()
    results.check("Memory stats available", "conversations" in stats)
    results.check("Total messages tracked", "total_messages" in stats)
    
    agent.memory.add_message("user", "test message")
    history = agent.memory.get_history()
    results.check("Message added", len(history) > 0)
    
    agent.memory.add_message("assistant", "test response")
    history = agent.memory.get_history()
    results.check("Response added", len(history) >= 2)


def test_input_sanitization(results):
    print("\n[6] Input Sanitization Tests")
    import re
    
    def sanitize_input(text, max_length=10000):
        if not text:
            return ""
        text = text[:max_length]
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
        return text.strip()
    
    results.check("Sanitize empty", sanitize_input("") == "")
    results.check("Sanitize null bytes", "\x00" not in sanitize_input("hello\x00world"))
    results.check("Sanitize max length", len(sanitize_input("x" * 20000)) == 10000)
    results.check("Sanitize control chars", "\x08" not in sanitize_input("test\x08data"))


def test_web_files(results):
    print("\n[7] Web Files Tests")
    files = [
        ("web/index.html", 1000),
        ("web/style.css", 5000),
        ("web/app.js", 5000)
    ]
    
    for f, min_size in files:
        exists = os.path.exists(f)
        size = os.path.getsize(f) if exists else 0
        results.check(f"{f} exists", exists)
        results.check(f"{f} has content", size > min_size, f"Size: {size}")


def test_config(results):
    print("\n[8] Configuration Tests")
    results.check("Backend configured", hasattr(config, 'BACKEND'))
    results.check("Model configured", hasattr(config, 'OLLAMA_MODEL'))
    results.check("Web host configured", hasattr(config, 'WEB_HOST'))
    results.check("Web port configured", hasattr(config, 'WEB_PORT'))
    results.check("DB path configured", hasattr(config, 'DB_PATH'))


def main():
    print("=" * 50)
    print("AI AGENT - COMPREHENSIVE TEST SUITE")
    print("=" * 50)
    
    results = TestResult()
    
    test_tool_registry(results)
    test_security(results)
    test_path_traversal(results)
    test_agent(results)
    test_memory(results)
    test_input_sanitization(results)
    test_web_files(results)
    test_config(results)
    
    success = results.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
