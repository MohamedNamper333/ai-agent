# AI Agent - خطة التطوير الشاملة

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** تطوير مشروع AI Agent Framework ليكون منافساً قوياً في كتابة الكود والتحليل والتخصص

**Architecture:** تحسين الجودة أولاً (اختبارات + UX + توثيق)، ثم التوسع (APIs خارجية + إشعارات + تعدد اللغات)، ثم التخصص (تحليل سياقي + multi-agent + أمان متقدم)

**Tech Stack:** Python 3.11+, FastAPI, Ollama, phi4:14b, JavaScript, CSS, HTML, SQLite (مستقبلاً)

---

## المرحلة 1: تحسين الجودة (1-2 أسبوع)

### Task 1: إضافة اختبارات FastAPI endpoints

**Files:**
- Create: `tests/test_web_endpoints.py`
- Modify: `web.py:465-475` (إضافة test client)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_web_endpoints.py
import pytest
from fastapi.testclient import TestClient
from web import app

client = TestClient(app)

def test_status_endpoint():
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert "model_loaded" in data
    assert "conversations" in data

def test_stats_endpoint():
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "tool_count" in data
    assert "fast_mode" in data

def test_settings_endpoint():
    response = client.get("/settings")
    assert response.status_code == 200
    data = response.json()
    assert "fast_mode" in data
    assert "rag_enabled" in data

def test_tools_endpoint():
    response = client.get("/tools")
    assert response.status_code == 200
    data = response.json()
    assert "tools" in data
    assert "total" in data
    assert "enabled" in data

def test_conversations_endpoint():
    response = client.get("/conversations")
    assert response.status_code == 200
    data = response.json()
    assert "conversations" in data
    assert "current" in data

def test_new_conversation():
    response = client.post("/conversations/new")
    assert response.status_code == 200
    data = response.json()
    assert "conversation_id" in data
    assert data["conversation_id"].startswith("conv_")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_web_endpoints.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'fastapi.testclient'"

- [ ] **Step 3: Install test dependency**

```bash
pip install httpx
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_web_endpoints.py -v`
Expected: PASS (all 6 tests)

- [ ] **Step 5: Commit**

```bash
git add tests/test_web_endpoints.py
git commit -m "test: add FastAPI endpoint tests for status, stats, settings, tools, conversations"
```

---

### Task 2: إضافة اختبارات Agent core

**Files:**
- Create: `tests/test_agent_core.py`
- Modify: `core/agent.py:581` (لا يحتاج تعديل - الاختبارات على الكود الموجود)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_agent_core.py
import pytest
from unittest.mock import Mock, MagicMock, patch
from core.agent import Agent, ToolCall, PlanStep, ExecutionPlan, TaskStatus

def test_agent_init():
    agent = Agent()
    assert agent is not None
    assert hasattr(agent, 'tools')
    assert hasattr(agent, 'memory')
    assert hasattr(agent, 'context')

def test_tool_call_dataclass():
    tc = ToolCall(name="test_tool", arguments={"key": "value"})
    assert tc.name == "test_tool"
    assert tc.arguments == {"key": "value"}
    assert tc.result is None
    assert tc.success is False

def test_plan_step_dataclass():
    step = PlanStep(description="Test step", tool="test_tool", args={"key": "value"})
    assert step.description == "Test step"
    assert step.tool == "test_tool"
    assert step.status == TaskStatus.PENDING

def test_execution_plan():
    plan = ExecutionPlan(goal="Test goal")
    assert plan.goal == "Test goal"
    assert plan.steps == []
    assert plan.status == TaskStatus.PENDING

def test_agent_has_retriever():
    agent = Agent()
    assert hasattr(agent, '_retriever')

def test_agent_has_fast_mode():
    agent = Agent()
    assert hasattr(agent, '_fast_mode')
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent_core.py -v`
Expected: FAIL with import errors

- [ ] **Step 3: Run test to verify it passes**

Run: `python -m pytest tests/test_agent_core.py -v`
Expected: PASS (all 6 tests)

- [ ] **Step 4: Commit**

```bash
git add tests/test_agent_core.py
git commit -m "test: add agent core tests for dataclasses and initialization"
```

---

### Task 3: إضافة اختبارات file_ops

**Files:**
- Create: `tests/test_file_ops.py`
- Modify: `tools/file_ops.py:191` (لا يحتاج تعديل)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_file_ops.py
import pytest
import os
import tempfile
from tools.file_ops import FileOps

@pytest.fixture
def file_ops():
    return FileOps()

@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

def test_file_ops_init():
    fo = FileOps()
    assert fo is not None
    assert hasattr(fo, 'read_file')
    assert hasattr(fo, 'write_file')
    assert hasattr(fo, 'edit_file')

def test_write_and_read_file(file_ops, temp_dir):
    test_file = os.path.join(temp_dir, "test.txt")
    result = file_ops.write_file(test_file, "Hello World")
    assert result.success is True
    
    result = file_ops.read_file(test_file)
    assert result.success is True
    assert "Hello World" in result.result

def test_list_directory(file_ops, temp_dir):
    # Create test files
    for i in range(3):
        with open(os.path.join(temp_dir, f"file{i}.txt"), "w") as f:
            f.write(f"content {i}")
    
    result = file_ops.list_directory(temp_dir)
    assert result.success is True
    assert "file0.txt" in result.result
    assert "file1.txt" in result.result
    assert "file2.txt" in result.result

def test_file_info(file_ops, temp_dir):
    test_file = os.path.join(temp_dir, "info.txt")
    with open(test_file, "w") as f:
        f.write("test content")
    
    result = file_ops.file_info(test_file)
    assert result.success is True
    assert "info.txt" in result.result

def test_glob_search(file_ops, temp_dir):
    for name in ["test1.py", "test2.py", "other.txt"]:
        with open(os.path.join(temp_dir, name), "w") as f:
            f.write("content")
    
    result = file_ops.glob_search(temp_dir, "*.py")
    assert result.success is True
    assert "test1.py" in result.result
    assert "test2.py" in result.result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_file_ops.py -v`
Expected: FAIL with import errors

- [ ] **Step 3: Run test to verify it passes**

Run: `python -m pytest tests/test_file_ops.py -v`
Expected: PASS (all 5 tests)

- [ ] **Step 4: Commit**

```bash
git add tests/test_file_ops.py
git commit -m "test: add file operations tests for read, write, list, info, glob"
```

---

### Task 4: إضافة اختبارات code_analysis

**Files:**
- Create: `tests/test_code_analysis.py`
- Modify: `tools/code_analysis.py:855` (لا يحتاج تعديل)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_code_analysis.py
import pytest
import tempfile
import os
from tools.code_analysis import CodeAnalysis

@pytest.fixture
def code_analysis():
    return CodeAnalysis()

@pytest.fixture
def temp_project():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test Python files
        with open(os.path.join(tmpdir, "main.py"), "w") as f:
            f.write("""
def hello():
    print("Hello World")

class Calculator:
    def add(self, a, b):
        return a + b
""")
        
        with open(os.path.join(tmpdir, "utils.py"), "w") as f:
            f.write("""
import os
import sys

def get_path():
    return os.getcwd()
""")
        
        yield tmpdir

def test_code_analysis_init():
    ca = CodeAnalysis()
    assert ca is not None
    assert hasattr(ca, 'analyze_project')
    assert hasattr(ca, 'code_review')

def test_analyze_project(code_analysis, temp_project):
    result = code_analysis.analyze_project(temp_project)
    assert result.success is True
    assert "main.py" in result.result
    assert "utils.py" in result.result

def test_code_review(code_analysis, temp_project):
    result = code_analysis.code_review(os.path.join(temp_project, "main.py"))
    assert result.success is True
    # Should contain some analysis
    assert len(result.result) > 0

def test_import_analysis(code_analysis, temp_project):
    result = code_analysis.import_analysis(os.path.join(temp_project, "utils.py"))
    assert result.success is True
    assert "os" in result.result
    assert "sys" in result.result

def test_complexity_metrics(code_analysis, temp_project):
    result = code_analysis.complexity_metrics(os.path.join(temp_project, "main.py"))
    assert result.success is True
    # Should contain metrics
    assert len(result.result) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_code_analysis.py -v`
Expected: FAIL with import errors

- [ ] **Step 3: Run test to verify it passes**

Run: `python -m pytest tests/test_code_analysis.py -v`
Expected: PASS (all 5 tests)

- [ ] **Step 4: Commit**

```bash
git add tests/test_code_analysis.py
git commit -m "test: add code analysis tests for project scan, review, imports, complexity"
```

---

### Task 5: إضافة اختبارات data_analysis

**Files:**
- Create: `tests/test_data_analysis.py`
- Modify: `tools/data_analysis.py:659` (لا يحتاج تعديل)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data_analysis.py
import pytest
import tempfile
import os
import json
from tools.data_analysis import DataAnalysis

@pytest.fixture
def data_analysis():
    return DataAnalysis()

@pytest.fixture
def temp_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test CSV
        csv_file = os.path.join(tmpdir, "test.csv")
        with open(csv_file, "w") as f:
            f.write("name,age,salary\n")
            f.write("Alice,30,50000\n")
            f.write("Bob,25,45000\n")
            f.write("Charlie,35,60000\n")
        
        # Create test JSON
        json_file = os.path.join(tmpdir, "test.json")
        with open(json_file, "w") as f:
            json.dump([
                {"name": "Alice", "age": 30},
                {"name": "Bob", "age": 25}
            ], f)
        
        yield tmpdir

def test_data_analysis_init():
    da = DataAnalysis()
    assert da is not None
    assert hasattr(da, 'analyze_file')
    assert hasattr(da, 'get_statistics')

def test_analyze_csv(data_analysis, temp_data):
    result = data_analysis.analyze_file(os.path.join(temp_data, "test.csv"))
    assert result.success is True
    assert "name" in result.result
    assert "age" in result.result
    assert "salary" in result.result

def test_analyze_json(data_analysis, temp_data):
    result = data_analysis.analyze_file(os.path.join(temp_data, "test.json"))
    assert result.success is True
    assert "name" in result.result
    assert "age" in result.result

def test_get_statistics(data_analysis, temp_data):
    result = data_analysis.get_statistics(os.path.join(temp_data, "test.csv"))
    assert result.success is True
    # Should contain statistical measures
    assert len(result.result) > 0

def test_data_quality(data_analysis, temp_data):
    result = data_analysis.data_quality(os.path.join(temp_data, "test.csv"))
    assert result.success is True
    # Should contain quality report
    assert len(result.result) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_data_analysis.py -v`
Expected: FAIL with import errors

- [ ] **Step 3: Run test to verify it passes**

Run: `python -m pytest tests/test_data_analysis.py -v`
Expected: PASS (all 5 tests)

- [ ] **Step 4: Commit**

```bash
git add tests/test_data_analysis.py
git commit -m "test: add data analysis tests for CSV, JSON, statistics, quality"
```

---

### Task 6: إضافة اختبارات scheduler

**Files:**
- Create: `tests/test_scheduler.py`
- Modify: `tools/scheduler.py:196` (لا يحتاج تعديل)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scheduler.py
import pytest
import time
from tools.scheduler import TaskScheduler

@pytest.fixture
def scheduler():
    return TaskScheduler()

def test_scheduler_init():
    s = TaskScheduler()
    assert s is not None
    assert hasattr(s, 'add_task')
    assert hasattr(s, 'remove_task')
    assert hasattr(s, 'list_tasks')

def test_add_task(scheduler):
    result = scheduler.add_task(
        name="test_task",
        task_type="interval",
        interval_seconds=60,
        command="echo test"
    )
    assert result is True
    tasks = scheduler.list_tasks()
    assert len(tasks) == 1
    assert tasks[0]["name"] == "test_task"

def test_remove_task(scheduler):
    scheduler.add_task(
        name="to_remove",
        task_type="interval",
        interval_seconds=60,
        command="echo test"
    )
    result = scheduler.remove_task("to_remove")
    assert result is True
    tasks = scheduler.list_tasks()
    assert len(tasks) == 0

def test_enable_disable_task(scheduler):
    scheduler.add_task(
        name="toggle_task",
        task_type="interval",
        interval_seconds=60,
        command="echo test"
    )
    
    result = scheduler.disable_task("toggle_task")
    assert result is True
    tasks = scheduler.list_tasks()
    assert tasks[0]["enabled"] is False
    
    result = scheduler.enable_task("toggle_task")
    assert result is True
    tasks = scheduler.list_tasks()
    assert tasks[0]["enabled"] is True

def test_task_history(scheduler):
    scheduler.add_task(
        name="history_task",
        task_type="interval",
        interval_seconds=60,
        command="echo test"
    )
    history = scheduler.get_task_history("history_task")
    assert isinstance(history, list)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_scheduler.py -v`
Expected: FAIL with import errors

- [ ] **Step 3: Run test to verify it passes**

Run: `python -m pytest tests/test_scheduler.py -v`
Expected: PASS (all 5 tests)

- [ ] **Step 4: Commit**

```bash
git add tests/test_scheduler.py
git commit -m "test: add scheduler tests for task CRUD, enable/disable, history"
```

---

### Task 7: إضافة اختبارات long_term_memory

**Files:**
- Create: `tests/test_long_term_memory.py`
- Modify: `tools/long_term_memory.py:112` (لا يحتاج تعديل)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_long_term_memory.py
import pytest
from tools.long_term_memory import LongTermMemory

@pytest.fixture
def ltm():
    return LongTermMemory()

def test_ltm_init():
    memory = LongTermMemory()
    assert memory is not None
    assert hasattr(memory, 'store_summary')
    assert hasattr(memory, 'search')

def test_store_and_search(ltm):
    ltm.store_summary(
        conversation_id="conv_123",
        summary="Discussion about Python programming",
        keywords=["python", "programming", "code"]
    )
    
    results = ltm.search("Python")
    assert len(results) > 0
    assert "conv_123" in str(results)

def test_search_no_results(ltm):
    results = ltm.search("nonexistent topic xyz")
    assert len(results) == 0

def test_multiple_summaries(ltm):
    ltm.store_summary("conv_1", "Python discussion", ["python"])
    ltm.store_summary("conv_2", "JavaScript discussion", ["javascript"])
    ltm.store_summary("conv_3", "Python advanced topics", ["python", "advanced"])
    
    results = ltm.search("Python")
    assert len(results) >= 2  # conv_1 and conv_3

def test_context_recall(ltm):
    ltm.store_summary("conv_1", "Machine learning project", ["ml", "ai"])
    context = ltm.recall_context("What was the ML project about?")
    assert context is not None
    assert len(context) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_long_term_memory.py -v`
Expected: FAIL with import errors

- [ ] **Step 3: Run test to verify it passes**

Run: `python -m pytest tests/test_long_term_memory.py -v`
Expected: PASS (all 5 tests)

- [ ] **Step 4: Commit**

```bash
git add tests/test_long_term_memory.py
git commit -m "test: add long-term memory tests for store, search, context recall"
```

---

### Task 8: تحسين Mobile UX - إضافة touch gestures

**Files:**
- Modify: `web/app.js:167-185` (إضافة swipe gesture)
- Modify: `web/style.css:1204-1231` (تحسين responsive)

- [ ] **Step 1: Add swipe gesture to close sidebar on mobile**

```javascript
// Add after initSidebarToggle function (line 185)
function initSwipeGesture() {
  let touchStartX = 0;
  let touchEndX = 0;
  
  document.addEventListener('touchstart', (e) => {
    touchStartX = e.changedTouches[0].screenX;
  }, false);
  
  document.addEventListener('touchend', (e) => {
    touchEndX = e.changedTouches[0].screenX;
    const diff = touchStartX - touchEndX;
    
    // Swipe left to close sidebar
    if (diff > 50 && appEl.classList.contains('sidebar-open-mobile')) {
      appEl.classList.remove('sidebar-open-mobile');
    }
    // Swipe right to open sidebar
    if (diff < -50 && !appEl.classList.contains('sidebar-open-mobile') && window.innerWidth <= 768) {
      appEl.classList.add('sidebar-open-mobile');
    }
  }, false);
}
```

- [ ] **Step 2: Call initSwipeGesture in DOMContentLoaded**

```javascript
// Add after initErrorBoundary() (line 45)
initSwipeGesture();
```

- [ ] **Step 3: Add CSS for better touch targets**

```css
/* Add to style.css after responsive section */
@media (max-width: 768px) {
  .conv-item { padding: 12px 10px; }
  .tree-item { padding: 10px 8px; }
  .tree-child { padding: 8px 8px; }
  .pill-sm { padding: 8px 16px; }
  .send-circle { width: 40px; height: 40px; }
  .mic-btn { width: 40px; height: 40px; }
}
```

- [ ] **Step 4: Run tests to verify nothing broke**

Run: `python -m pytest tests/ -v`
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add web/app.js web/style.css
git commit -m "feat: add swipe gestures and improved touch targets for mobile"
```

---

### Task 9: إنشاء System Design Documentation

**Files:**
- Create: `docs/ARCHITECTURE.md`
- Create: `docs/API_REFERENCE.md`

- [ ] **Step 1: Create ARCHITECTURE.md**

```markdown
# AI Agent Architecture

## System Overview

The AI Agent is a modular, plugin-based framework for building intelligent assistants.

## Core Components

### 1. Agent Core (`core/agent.py`)
- **Responsibility:** Main orchestration loop
- **Key Classes:** Agent, ToolCall, PlanStep, ExecutionPlan
- **Flow:** Plan → Execute → Reflect → Respond

### 2. Tool Registry (`core/tools.py`)
- **Responsibility:** Tool management and execution
- **Key Classes:** Tool, ToolResult, ToolRegistry
- **Features:** Lazy loading, category management, parallel execution

### 3. Memory System (`core/memory.py`)
- **Responsibility:** Conversation persistence and context
- **Key Classes:** ConversationMemory, Message
- **Features:** Multi-conversation, token-based trimming, batch saves

### 4. RAG Pipeline (`rag/`)
- **Responsibility:** Document retrieval and augmentation
- **Components:** Retriever, VectorStore, Embedder
- **Features:** Hybrid search (semantic + BM25), chunking

### 5. Web Interface (`web/`)
- **Responsibility:** User interaction
- **Components:** FastAPI server, SPA frontend
- **Features:** SSE streaming, dark mode, mobile responsive

## Data Flow

```
User Input → Agent → Tool Selection → Execution → Response
                    ↓
              Memory Update → RAG Query → Enhanced Context
```

## Security Model

- Path traversal prevention
- Input sanitization
- Rate limiting (per-role)
- API key authentication
- Sandboxed code execution
```

- [ ] **Step 2: Create API_REFERENCE.md**

```markdown
# API Reference

## Endpoints

### Chat
- `POST /chat` - Send message and receive streaming response
- Body: `{ conversation_id, message, stream, use_rag }`

### Conversations
- `GET /conversations` - List all conversations
- `POST /conversations/new` - Create new conversation
- `GET /conversations/{id}` - Get conversation history
- `DELETE /conversations/{id}` - Delete conversation

### Settings
- `GET /settings` - Get current settings
- `POST /settings/fast-mode` - Toggle fast mode
- `POST /settings/rag` - Toggle RAG

### Tools
- `GET /tools` - List all tools with status
- `POST /tools/{name}/enable` - Enable tool
- `POST /tools/{name}/disable` - Disable tool
- `POST /tools/category/{cat}/enable` - Enable category
- `POST /tools/category/{cat}/disable` - Disable category

### Auth
- `POST /auth/register` - Register user
- `GET /auth/users` - List users (admin only)

### Upload
- `POST /upload` - Upload document for RAG indexing
```

- [ ] **Step 3: Commit**

```bash
git add docs/ARCHITECTURE.md docs/API_REFERENCE.md
git commit -m "docs: add architecture and API reference documentation"
```

---

## المرحلة 2: التوسع (2-4 أسابيع)

### Task 10: تكامل DuckDuckGo للبحث الخارجي

**Files:**
- Modify: `tools/web_search.py:236` (تحسين search)
- Create: `tests/test_web_search_duckduckgo.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_web_search_duckduckgo.py
import pytest
from tools.web_search import WebTools

@pytest.fixture
def web_tools():
    return WebTools()

def test_web_tools_init():
    wt = WebTools()
    assert wt is not None
    assert hasattr(wt, 'search_web')
    assert hasattr(wt, 'fetch_url')

def test_search_web_duckduckgo(web_tools):
    result = web_tools.search_web("Python programming", num_results=3)
    assert result.success is True
    # Should contain search results
    assert len(result.result) > 0

def test_fetch_url(web_tools):
    result = web_tools.fetch_url("https://httpbin.org/get")
    assert result.success is True
    assert "headers" in result.result.lower() or "args" in result.result.lower()

def test_fetch_json(web_tools):
    result = web_tools.fetch_json("https://httpbin.org/json")
    assert result.success is True
    # Should be valid JSON-like content
    assert len(result.result) > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_web_search_duckduckgo.py -v`
Expected: FAIL (search not implemented properly)

- [ ] **Step 3: Implement DuckDuckGo search**

```python
# Add to tools/web_search.py
def search_web(self, query: str, num_results: int = 5) -> ToolResult:
    """Search the web using DuckDuckGo HTML scraping"""
    try:
        import requests
        from bs4 import BeautifulSoup
        
        url = f"https://html.duckduckgo.com/html/?q={query}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        results = []
        for result in soup.find_all('div', class_='result')[:num_results]:
            title_elem = result.find('a', class_='result__a')
            snippet_elem = result.find('a', class_='result__snippet')
            
            if title_elem:
                title = title_elem.get_text(strip=True)
                link = title_elem.get('href', '')
                snippet = snippet_elem.get_text(strip=True) if snippet_elem else ''
                
                results.append({
                    'title': title,
                    'url': link,
                    'snippet': snippet
                })
        
        return ToolResult(
            success=True,
            result=json.dumps(results, indent=2),
            tool="search_web"
        )
    except Exception as e:
        return ToolResult(
            success=False,
            result=f"Search failed: {str(e)}",
            tool="search_web"
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_web_search_duckduckgo.py -v`
Expected: PASS (all 4 tests)

- [ ] **Step 5: Commit**

```bash
git add tools/web_search.py tests/test_web_search_duckduckgo.py
git commit -m "feat: add DuckDuckGo web search integration"
```

---

### Task 11: تحسين نظام الإشعارات

**Files:**
- Create: `core/notifications.py`
- Modify: `web.py:486` (إضافة notification endpoints)
- Create: `tests/test_notifications.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_notifications.py
import pytest
from core.notifications import NotificationManager

@pytest.fixture
def nm():
    return NotificationManager()

def test_notification_manager_init():
    manager = NotificationManager()
    assert manager is not None
    assert hasattr(manager, 'add_notification')
    assert hasattr(manager, 'get_notifications')
    assert hasattr(manager, 'mark_read')

def test_add_notification(nm):
    result = nm.add_notification(
        user_id="user1",
        title="Test Notification",
        message="This is a test",
        type="info"
    )
    assert result is True

def test_get_notifications(nm):
    nm.add_notification("user1", "Title 1", "Message 1", "info")
    nm.add_notification("user1", "Title 2", "Message 2", "warning")
    
    notifications = nm.get_notifications("user1")
    assert len(notifications) == 2
    assert notifications[0]["title"] == "Title 1"

def test_mark_read(nm):
    nm.add_notification("user1", "Title", "Message", "info")
    notifications = nm.get_notifications("user1")
    notif_id = notifications[0]["id"]
    
    result = nm.mark_read(notif_id)
    assert result is True
    
    notifications = nm.get_notifications("user1", unread_only=True)
    assert len(notifications) == 0

def test_notification_types(nm):
    for ntype in ["info", "warning", "error", "success"]:
        nm.add_notification("user1", f"Type {ntype}", "Message", ntype)
    
    notifications = nm.get_notifications("user1")
    assert len(notifications) == 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_notifications.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Create notifications module**

```python
# core/notifications.py
import json
import uuid
from datetime import datetime
from typing import List, Optional, Dict
from pathlib import Path

class Notification:
    def __init__(self, user_id: str, title: str, message: str, ntype: str = "info"):
        self.id = str(uuid.uuid4())[:8]
        self.user_id = user_id
        self.title = title
        self.message = message
        self.type = ntype
        self.read = False
        self.created_at = datetime.now().isoformat()
    
    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "message": self.message,
            "type": self.type,
            "read": self.read,
            "created_at": self.created_at
        }

class NotificationManager:
    def __init__(self, storage_path: str = "notifications.json"):
        self.storage_path = Path(storage_path)
        self.notifications: List[Notification] = []
        self._load()
    
    def _load(self):
        if self.storage_path.exists():
            try:
                data = json.loads(self.storage_path.read_text(encoding="utf-8"))
                for item in data:
                    notif = Notification(
                        item["user_id"],
                        item["title"],
                        item["message"],
                        item["type"]
                    )
                    notif.id = item["id"]
                    notif.read = item.get("read", False)
                    notif.created_at = item.get("created_at", datetime.now().isoformat())
                    self.notifications.append(notif)
            except Exception:
                self.notifications = []
    
    def _save(self):
        data = [n.to_dict() for n in self.notifications]
        self.storage_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
    
    def add_notification(self, user_id: str, title: str, message: str, ntype: str = "info") -> bool:
        notif = Notification(user_id, title, message, ntype)
        self.notifications.append(notif)
        self._save()
        return True
    
    def get_notifications(self, user_id: str, unread_only: bool = False) -> List[Dict]:
        result = []
        for n in self.notifications:
            if n.user_id == user_id:
                if unread_only and n.read:
                    continue
                result.append(n.to_dict())
        return result
    
    def mark_read(self, notification_id: str) -> bool:
        for n in self.notifications:
            if n.id == notification_id:
                n.read = True
                self._save()
                return True
        return False
    
    def get_unread_count(self, user_id: str) -> int:
        return len([n for n in self.notifications if n.user_id == user_id and not n.read])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_notifications.py -v`
Expected: PASS (all 5 tests)

- [ ] **Step 5: Commit**

```bash
git add core/notifications.py tests/test_notifications.py
git commit -m "feat: add notification manager with CRUD and read tracking"
```

---

### Task 12: إضافة دعم تعدد اللغات في الردود

**Files:**
- Modify: `core/context.py:116` (تحسين system prompt)
- Create: `core/language_detector.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_language.py
import pytest
from core.language_detector import detect_language, get_language_name

def test_detect_english():
    lang = detect_language("Hello, how are you?")
    assert lang == "en"

def test_detect_arabic():
    lang = detect_language("مرحبا، كيف حالك؟")
    assert lang == "ar"

def test_detect_spanish():
    lang = detect_language("Hola, como estas?")
    assert lang == "es"

def test_detect_french():
    lang = detect_language("Bonjour, comment allez-vous?")
    assert lang == "fr"

def test_get_language_name():
    assert get_language_name("en") == "English"
    assert get_language_name("ar") == "Arabic"
    assert get_language_name("es") == "Spanish"

def test_unknown_language():
    lang = detect_language("12345 !@#$%")
    assert lang == "en"  # Default to English
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_language.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Create language detector**

```python
# core/language_detector.py
import re
from typing import Dict

# Unicode ranges for different scripts
LANGUAGE_PATTERNS: Dict[str, str] = {
    "ar": r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]',
    "he": r'[\u0590-\u05FF\uFB1D-\uFB4F]',
    "zh": r'[\u4E00-\u9FFF\u3400-\u4DBF]',
    "ja": r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]',
    "ko": r'[\uAC00-\uD7AF\u1100-\u11FF]',
    "hi": r'[\u0900-\u097F]',
    "th": r'[\u0E00-\u0E7F]',
    "ru": r'[\u0400-\u04FF]',
}

LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English",
    "ar": "Arabic",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "ru": "Russian",
    "zh": "Chinese",
    "ja": "Japanese",
    "ko": "Korean",
    "hi": "Hindi",
    "th": "Thai",
    "he": "Hebrew",
    "tr": "Turkish",
    "nl": "Dutch",
    "sv": "Swedish",
    "pl": "Polish",
}

def detect_language(text: str) -> str:
    """Detect the primary language of text using Unicode ranges and heuristics"""
    if not text or not text.strip():
        return "en"
    
    # Check for non-Latin scripts first
    for lang, pattern in LANGUAGE_PATTERNS.items():
        if re.search(pattern, text):
            return lang
    
    # For Latin scripts, use word frequency analysis
    text_lower = text.lower()
    
    # Common English words
    english_words = ['the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                     'could', 'should', 'may', 'might', 'can', 'shall', 'i', 'you',
                     'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them']
    
    # Common Spanish words
    spanish_words = ['el', 'la', 'los', 'las', 'un', 'una', 'de', 'del', 'en', 'con',
                     'por', 'para', 'como', 'pero', 'mas', 'este', 'esta', 'eso', 'fue']
    
    # Common French words
    french_words = ['le', 'la', 'les', 'un', 'une', 'de', 'du', 'des', 'en', 'dans',
                    'avec', 'pour', 'comme', 'mais', 'plus', 'ce', 'cette', 'qui', 'que']
    
    # Common German words
    german_words = ['der', 'die', 'das', 'ein', 'eine', 'von', 'in', 'mit', 'auf',
                    'fur', 'aber', 'auch', 'als', 'noch', 'nur', 'schon', 'wenn', 'ich']
    
    # Count matches
    words = re.findall(r'\b\w+\b', text_lower)
    
    en_count = sum(1 for w in words if w in english_words)
    es_count = sum(1 for w in words if w in spanish_words)
    fr_count = sum(1 for w in words if w in french_words)
    de_count = sum(1 for w in words if w in german_words)
    
    counts = {'en': en_count, 'es': es_count, 'fr': fr_count, 'de': de_count}
    max_lang = max(counts, key=counts.get)
    
    # If no strong signal, default to English
    if counts[max_lang] == 0:
        return "en"
    
    return max_lang

def get_language_name(code: str) -> str:
    """Get full language name from ISO code"""
    return LANGUAGE_NAMES.get(code, "Unknown")

def get_supported_languages() -> Dict[str, str]:
    """Get all supported languages"""
    return LANGUAGE_NAMES.copy()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_language.py -v`
Expected: PASS (all 6 tests)

- [ ] **Step 5: Commit**

```bash
git add core/language_detector.py tests/test_language.py
git commit -m "feat: add language detection for multilingual support"
```

---

## المرحلة 3: التخصص (4+ أسابيع)

### Task 13: تحليل سياقي متقدم

**Files:**
- Create: `core/context_analyzer.py`
- Modify: `core/agent.py:581` (تكامل مع analyzer)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_context_analyzer.py
import pytest
from core.context_analyzer import ContextAnalyzer

@pytest.fixture
def analyzer():
    return ContextAnalyzer()

def test_analyzer_init():
    analyzer = ContextAnalyzer()
    assert analyzer is not None
    assert hasattr(analyzer, 'analyze_intent')
    assert hasattr(analyzer, 'extract_entities')

def test_analyze_intent_code(analyzer):
    result = analyzer.analyze_intent("Write a Python function to sort a list")
    assert result["intent"] == "code_generation"
    assert result["confidence"] > 0.5

def test_analyze_intent_analysis(analyzer):
    result = analyzer.analyze_intent("Analyze the performance of this code")
    assert result["intent"] == "code_analysis"
    assert result["confidence"] > 0.5

def test_analyze_intent_search(analyzer):
    result = analyzer.analyze_intent("Search for latest AI news")
    assert result["intent"] == "web_search"
    assert result["confidence"] > 0.5

def test_extract_entities(analyzer):
    result = analyzer.extract_entities("Read the file data.csv and analyze it")
    assert "file_path" in result
    assert "data.csv" in result["file_path"]

def test_context_score(analyzer):
    score = analyzer.calculate_relevance_score(
        "Python programming",
        "This document discusses Python best practices"
    )
    assert 0 <= score <= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_context_analyzer.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Create context analyzer**

```python
# core/context_analyzer.py
import re
from typing import Dict, List, Tuple

class ContextAnalyzer:
    def __init__(self):
        self.intent_patterns = {
            "code_generation": [
                r"write\s+(a\s+)?(python|javascript|java|c\+\+|ruby|go|rust|swift)",
                r"create\s+(a\s+)?(function|class|module|script)",
                r"implement\s+(a\s+)?",
                r"build\s+(a\s+)?(tool|utility|helper)",
                r"generate\s+(a\s+)?(code|script|program)",
            ],
            "code_analysis": [
                r"analyze\s+(the\s+)?(code|performance|complexity)",
                r"review\s+(this\s+)?(code|pull request)",
                r"check\s+(for\s+)?(bugs|errors|issues)",
                r"optimize\s+(this\s+)?",
                r"refactor\s+(this\s+)?",
            ],
            "web_search": [
                r"search\s+(for\s+)?",
                r"find\s+(me\s+)?",
                r"look\s+up",
                r"google\s+",
                r"what\s+(is|are|was|were)\s+",
                r"latest\s+(news|trends|updates)",
            ],
            "file_operation": [
                r"read\s+(the\s+)?file",
                r"write\s+(to\s+)?file",
                r"edit\s+(the\s+)?file",
                r"delete\s+(the\s+)?file",
                r"create\s+(a\s+)?file",
            ],
            "data_analysis": [
                r"analyze\s+(the\s+)?data",
                r"show\s+(me\s+)?(statistics|stats|metrics)",
                r"create\s+(a\s+)?(chart|graph|visualization)",
                r"calculate\s+(the\s+)?",
                r"summarize\s+(this\s+)?",
            ],
            "explanation": [
                r"explain\s+(this|how|what|why)",
                r"what\s+(does|is|are)\s+",
                r"how\s+(does|do|can|should)\s+",
                r"tell\s+me\s+about",
                r"describe\s+",
            ],
        }
        
        self.entity_patterns = {
            "file_path": [
                r'(?:[\w\/\\.-]+\/)*[\w.-]+\.(?:py|js|ts|java|cpp|c|rs|go|rb|php|html|css|json|xml|csv|txt|md)',
                r'(?:C:|D:|\/home|\/usr|\/etc)\\?[\/\w.-]+',
            ],
            "programming_language": [
                r'\b(python|javascript|java|typescript|c\+\+|c#|ruby|go|rust|swift|kotlin|php|scala|r|matlab|sql)\b',
            ],
            "framework": [
                r'\b(django|flask|fastapi|react|vue|angular|node\.js|express|spring\.net|rails|laravel)\b',
            ],
            "concept": [
                r'\b(algorithm|data structure|design pattern|api|database|cache|queue|stack|tree|graph)\b',
            ],
        }
    
    def analyze_intent(self, text: str) -> Dict[str, any]:
        """Analyze the intent of user input"""
        text_lower = text.lower()
        scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    score += 1
            scores[intent] = score
        
        if max(scores.values()) == 0:
            return {"intent": "general", "confidence": 0.5}
        
        best_intent = max(scores, key=scores.get)
        confidence = min(scores[best_intent] / 3, 1.0)  # Normalize to 0-1
        
        return {
            "intent": best_intent,
            "confidence": confidence,
            "all_scores": scores
        }
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        entities = {}
        
        for entity_type, patterns in self.entity_patterns.items():
            found = []
            for pattern in patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                found.extend(matches)
            if found:
                entities[entity_type] = list(set(found))
        
        return entities
    
    def calculate_relevance_score(self, query: str, context: str) -> float:
        """Calculate relevance score between query and context"""
        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        context_words = set(re.findall(r'\b\w+\b', context.lower()))
        
        if not query_words:
            return 0.0
        
        intersection = query_words & context_words
        score = len(intersection) / len(query_words)
        
        return min(score * 1.5, 1.0)  # Boost and cap at 1.0
    
    def get_suggested_tools(self, intent: str, entities: Dict) -> List[str]:
        """Suggest tools based on intent and entities"""
        tool_suggestions = {
            "code_generation": ["run_code", "write_file"],
            "code_analysis": ["code_review", "complexity_metrics", "import_analysis"],
            "web_search": ["search_web", "fetch_url"],
            "file_operation": ["read_file", "write_file", "edit_file", "list_directory"],
            "data_analysis": ["analyze_file", "get_statistics", "create_chart"],
            "explanation": ["code_review"],
        }
        
        tools = tool_suggestions.get(intent, [])
        
        # Add entity-based suggestions
        if "programming_language" in entities:
            tools.append("run_code")
        
        if "file_path" in entities:
            tools.extend(["read_file", "file_info"])
        
        return list(set(tools))  # Remove duplicates
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_context_analyzer.py -v`
Expected: PASS (all 6 tests)

- [ ] **Step 5: Commit**

```bash
git add core/context_analyzer.py tests/test_context_analyzer.py
git commit -m "feat: add advanced context analyzer with intent detection and entity extraction"
```

---

### Task 14: تحسين Multi-Agent Collaboration

**Files:**
- Modify: `tools/multi_agent.py:272` (تحسين الـ debate)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_multi_agent_enhanced.py
import pytest
from unittest.mock import Mock, MagicMock, patch
from tools.multi_agent import MultiAgentOrchestrator

@pytest.fixture
def orchestrator():
    return MultiAgentOrchestrator()

def test_orchestrator_init():
    orch = MultiAgentOrchestrator()
    assert orch is not None
    assert hasattr(orch, 'run_council')
    assert hasattr(orch, 'delegate_task')

def test_council_mode_sequential(orchestrator):
    with patch.object(orchestrator, '_call_llm') as mock_llm:
        mock_llm.return_value = "Test response"
        result = orchestrator.run_council("Test topic", mode="sequential")
        assert result is not None
        assert len(result) > 0

def test_agent_specializations():
    from tools.multi_agent import AnalystAgent, ProgrammerAgent, ReviewerAgent, ArchitectAgent
    
    analyst = AnalystAgent()
    programmer = ProgrammerAgent()
    reviewer = ReviewerAgent()
    architect = ArchitectAgent()
    
    assert analyst.role == "analyst"
    assert programmer.role == "programmer"
    assert reviewer.role == "reviewer"
    assert architect.role == "architect"

def test_consensus_voting():
    from tools.multi_agent import MultiAgentOrchestrator
    
    orch = MultiAgentOrchestrator()
    votes = {
        "analyst": {"confidence": 0.8, "suggestion": "Option A"},
        "programmer": {"confidence": 0.7, "suggestion": "Option A"},
        "reviewer": {"confidence": 0.9, "suggestion": "Option B"},
        "architect": {"confidence": 0.6, "suggestion": "Option A"},
    }
    
    result = orch._calculate_consensus(votes)
    assert "winner" in result
    assert "confidence" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_multi_agent_enhanced.py -v`
Expected: FAIL

- [ ] **Step 3: Implement enhancements**

```python
# Add to tools/multi_agent.py

class AnalystAgent:
    def __init__(self):
        self.role = "analyst"
        self.system_prompt = """You are an analyst agent specializing in:
- Requirements analysis
- Data gathering
- Problem decomposition
- Risk assessment
Provide structured analysis with clear findings."""
    
    def analyze(self, topic: str) -> dict:
        return {
            "role": self.role,
            "analysis": f"Analyzing: {topic}",
            "confidence": 0.8,
            "findings": []
        }

class ProgrammerAgent:
    def __init__(self):
        self.role = "programmer"
        self.system_prompt = """You are a programmer agent specializing in:
- Code implementation
- Algorithm design
- Performance optimization
- Technical solutions
Provide code examples and technical details."""
    
    def analyze(self, topic: str) -> dict:
        return {
            "role": self.role,
            "analysis": f"Technical approach: {topic}",
            "confidence": 0.85,
            "code_suggestions": []
        }

class ReviewerAgent:
    def __init__(self):
        self.role = "reviewer"
        self.system_prompt = """You are a reviewer agent specializing in:
- Code review
- Quality assurance
- Best practices
- Security review
Provide constructive feedback and improvements."""
    
    def analyze(self, topic: str) -> dict:
        return {
            "role": self.role,
            "analysis": f"Review of: {topic}",
            "confidence": 0.75,
            "suggestions": []
        }

class ArchitectAgent:
    def __init__(self):
        self.role = "architect"
        self.system_prompt = """You are an architect agent specializing in:
- System design
- Architecture patterns
- Scalability
- Integration
Provide high-level design recommendations."""
    
    def analyze(self, topic: str) -> dict:
        return {
            "role": self.role,
            "analysis": f"Architecture for: {topic}",
            "confidence": 0.7,
            "design_recommendations": []
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_multi_agent_enhanced.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tools/multi_agent.py tests/test_multi_agent_enhanced.py
git commit -m "feat: enhance multi-agent with specialized agent classes"
```

---

### Task 15: نظام أمان متقدم مع تدقيقات تلقائية

**Files:**
- Create: `core/security_scanner.py`
- Modify: `core/tools.py:1006` (تكامل مع scanner)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_security_scanner.py
import pytest
from core.security_scanner import SecurityScanner

@pytest.fixture
def scanner():
    return SecurityScanner()

def test_scanner_init():
    scanner = SecurityScanner()
    assert scanner is not None
    assert hasattr(scanner, 'scan_code')
    assert hasattr(scanner, 'scan_file')

def test_scan_code_sql_injection(scanner):
    result = scanner.scan_code("SELECT * FROM users WHERE id = " + user_input)
    assert result["risk_level"] == "high"
    assert len(result["issues"]) > 0

def test_scan_code_xss(scanner):
    code = '<div>' + user_input + '</div>'
    result = scanner.scan_code(code)
    assert result["risk_level"] in ["medium", "high"]

def test_scan_code_safe(scanner):
    code = "def add(a, b): return a + b"
    result = scanner.scan_code(code)
    assert result["risk_level"] == "low"
    assert len(result["issues"]) == 0

def test_scan_code_hardcoded_secrets(scanner):
    code = 'password = "secret123"\napi_key = "sk-1234567890"'
    result = scanner.scan_code(code)
    assert result["risk_level"] in ["medium", "high"]
    assert any("secret" in issue.lower() or "credential" in issue.lower() for issue in result["issues"])

def test_get_recommendations(scanner):
    result = scanner.get_recommendations("high")
    assert len(result) > 0
    assert all(isinstance(r, str) for r in result)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_security_scanner.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Create security scanner**

```python
# core/security_scanner.py
import re
from typing import Dict, List, Tuple

class SecurityScanner:
    def __init__(self):
        self.patterns = {
            "sql_injection": {
                "patterns": [
                    r"SELECT\s+.*FROM\s+.*WHERE\s+.*\+",
                    r"INSERT\s+INTO\s+.*VALUES\s+.*\+",
                    r"UPDATE\s+.*SET\s+.*WHERE\s+.*\+",
                    r"DELETE\s+FROM\s+.*WHERE\s+.*\+",
                    r"execute\s*\(.*['\"].*['\"]",
                ],
                "risk_level": "high",
                "description": "Potential SQL injection vulnerability"
            },
            "xss": {
                "patterns": [
                    r"innerHTML\s*=\s*.*\+",
                    r"document\.write\s*\(.*\+",
                    r"\.html\s*\(.*\+",
                    r"eval\s*\(.*\+",
                ],
                "risk_level": "high",
                "description": "Potential XSS vulnerability"
            },
            "hardcoded_secrets": {
                "patterns": [
                    r"password\s*=\s*['\"][^'\"]+['\"]",
                    r"api_key\s*=\s*['\"][^'\"]+['\"]",
                    r"secret\s*=\s*['\"][^'\"]+['\"]",
                    r"token\s*=\s*['\"][^'\"]+['\"]",
                    r"sk-[a-zA-Z0-9]{20,}",
                ],
                "risk_level": "high",
                "description": "Hardcoded secret or credential"
            },
            "insecure_random": {
                "patterns": [
                    r"random\.random\s*\(\)",
                    r"Math\.random\s*\(\)",
                ],
                "risk_level": "medium",
                "description": "Insecure random number generation"
            },
            "eval_usage": {
                "patterns": [
                    r"eval\s*\(",
                    r"exec\s*\(",
                    r"compile\s*\(.*['\"]exec['\"]",
                ],
                "risk_level": "high",
                "description": "Dynamic code execution (potential code injection)"
            },
            "path_traversal": {
                "patterns": [
                    r"\.\.\/",
                    r"\.\.\\",
                    r"open\s*\(.*\.\.",
                ],
                "risk_level": "medium",
                "description": "Potential path traversal"
            },
            "insecure_deserialization": {
                "patterns": [
                    r"pickle\.loads?\s*\(",
                    r"yaml\.load\s*\(",
                    r"marshal\.loads?\s*\(",
                ],
                "risk_level": "high",
                "description": "Insecure deserialization"
            },
            "debug_code": {
                "patterns": [
                    r"print\s*\(.*debug",
                    r"console\.log\s*\(",
                    r"import\s+pdb",
                    r"breakpoint\s*\(\)",
                ],
                "risk_level": "low",
                "description": "Debug code in production"
            },
        }
        
        self.recommendations = {
            "high": [
                "Use parameterized queries for database operations",
                "Sanitize and validate all user inputs",
                "Use environment variables for secrets",
                "Avoid eval() and exec() - use safer alternatives",
                "Use secure deserialization libraries",
            ],
            "medium": [
                "Use cryptographically secure random generators",
                "Validate file paths to prevent traversal",
                "Use context managers for file operations",
                "Add input length limits",
            ],
            "low": [
                "Remove debug statements before production",
                "Use proper logging instead of print",
                "Configure logging levels appropriately",
            ],
        }
    
    def scan_code(self, code: str) -> Dict:
        """Scan code for security issues"""
        issues = []
        risk_scores = {"low": 0, "medium": 0, "high": 0}
        
        for category, config in self.patterns.items():
            for pattern in config["patterns"]:
                if re.search(pattern, code, re.IGNORECASE):
                    issues.append(f"{config['description']} ({category})")
                    risk_scores[config["risk_level"]] += 1
                    break  # One match per category is enough
        
        # Determine overall risk level
        if risk_scores["high"] > 0:
            risk_level = "high"
        elif risk_scores["medium"] > 0:
            risk_level = "medium"
        elif risk_scores["low"] > 0:
            risk_level = "low"
        else:
            risk_level = "low"
        
        return {
            "risk_level": risk_level,
            "issues": issues,
            "issue_count": len(issues),
            "risk_scores": risk_scores
        }
    
    def scan_file(self, file_path: str) -> Dict:
        """Scan a file for security issues"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
            
            result = self.scan_code(code)
            result["file_path"] = file_path
            return result
        except Exception as e:
            return {
                "risk_level": "unknown",
                "issues": [f"Failed to scan file: {str(e)}"],
                "issue_count": 0,
                "file_path": file_path
            }
    
    def get_recommendations(self, risk_level: str) -> List[str]:
        """Get security recommendations based on risk level"""
        if risk_level == "high":
            return self.recommendations["high"] + self.recommendations["medium"]
        elif risk_level == "medium":
            return self.recommendations["medium"]
        else:
            return self.recommendations["low"]
    
    def generate_report(self, scan_result: Dict) -> str:
        """Generate a human-readable security report"""
        report_lines = [
            f"Security Report",
            f"================",
            f"Risk Level: {scan_result['risk_level'].upper()}",
            f"Issues Found: {scan_result['issue_count']}",
            ""
        ]
        
        if scan_result["issues"]:
            report_lines.append("Issues:")
            for i, issue in enumerate(scan_result["issues"], 1):
                report_lines.append(f"  {i}. {issue}")
            report_lines.append("")
        
        recommendations = self.get_recommendations(scan_result["risk_level"])
        report_lines.append("Recommendations:")
        for rec in recommendations:
            report_lines.append(f"  - {rec}")
        
        return "\n".join(report_lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_security_scanner.py -v`
Expected: PASS (all 6 tests)

- [ ] **Step 5: Commit**

```bash
git add core/security_scanner.py tests/test_security_scanner.py
git commit -m "feat: add advanced security scanner with pattern detection"
```

---

## ملخص المراحل

### المرحلة 1: تحسين الجودة (1-2 أسبوع)
- [ ] Task 1: اختبارات FastAPI endpoints
- [ ] Task 2: اختبارات Agent core
- [ ] Task 3: اختبارات file_ops
- [ ] Task 4: اختبارات code_analysis
- [ ] Task 5: اختبارات data_analysis
- [ ] Task 6: اختبارات scheduler
- [ ] Task 7: اختبارات long_term_memory
- [ ] Task 8: تحسين Mobile UX
- [ ] Task 9: توثيق النظام

### المرحلة 2: التوسع (2-4 أسابيع)
- [ ] Task 10: تكامل DuckDuckGo
- [ ] Task 11: نظام الإشعارات
- [ ] Task 12: دعم تعدد اللغات

### المرحلة 3: التخصص (4+ أسابيع)
- [ ] Task 13: تحليل سياقي متقدم
- [ ] Task 14: تحسين Multi-Agent
- [ ] Task 15: نظام أمان متقدم

---

## معايير النجاح

### المرحلة 1
- تغطية الاختبارات > 80% للوحدات الحرجة
- وقت استجابة الجوال < 100ms
- وثائق مفصلة لكل المكونات

### المرحلة 2
- بحث خارجي يعمل مع 3+ مصادر
- إشعارات فورية مع < 1s latency
- دعم 5+ لغات مع كشف تلقائي

### المرحلة 3
- كشف السياق بدقة > 85%
- تعاون Multi-Agent مع 4+ وكلاء
- فحص أمني مع < 5% false positives

---

**Plan complete and saved to `docs/plans/2026-05-31-comprehensive-development-plan.md`. Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
