# AI Agent - Advanced AI Assistant

Multi-model AI agent with **58+ tools**, hybrid RAG search, streaming web interface, multi-agent council, Docker sandbox, long-term memory, and plugin system. Supports Ollama, GPT4All, and direct GGUF backends.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [LLM Backends](#llm-backends)
- [Web Interface](#web-interface)
- [API Endpoints](#api-endpoints)
- [CLI Mode](#cli-mode)
- [Tool Categories](#tool-categories)
- [Plugin System](#plugin-system)
- [RAG Engine](#rag-engine)
- [Security](#security)
- [Testing](#testing)
- [Benchmarks](#benchmarks)
- [Improvement Log](#improvement-log)
- [Project Structure](#project-structure)
- [License](#license)

---

## Features

| Feature | Description |
|---------|-------------|
| **58+ Tools** | File ops, web search, Git, code analysis, data analysis, documents, voice, scheduling, Docker |
| **Multi-Model** | Ollama, GPT4All, direct GGUF (llama-cpp-python), OpenCode Zen |
| **Hybrid RAG** | Semantic vector search + BM25 keyword retrieval |
| **Streaming Chat** | Real-time SSE streaming with tool call visualization |
| **Multi-Agent Council** | Decompose complex problems across specialist agents |
| **Docker Sandbox** | Secure code execution in isolated containers |
| **Long-Term Memory** | Persistent conversation store with auto-summarization |
| **Plugin System** | Extend with custom tools via simple Python API |
| **Reasoning Engine** | Chain-of-Thought (CoT) reasoning for complex tasks |
| **Task Scheduling** | Cron-like scheduler for recurring tasks |
| **Voice I/O** | Text-to-speech (gTTS) and speech recognition |
| **Security** | Path traversal protection, input sanitization, rate limiting, auth |
| **Bilingual** | Full Arabic + English support in system prompt and responses |

---

## Architecture

```
ai-agent/
├── main.py                  # CLI entry point (argparse)
├── web.py                   # FastAPI server (381 lines)
├── config.py                # Central config loader
├── config.txt               # User-editable settings
├── config/
│   └── env_loader.py        # .env file loader with caching
│
├── core/                    # Core engine
│   ├── agent.py             # Main agent loop
│   ├── tools.py             # Tool registry & discovery
│   ├── memory.py            # Conversation memory
│   ├── context.py           # Context window management
│   ├── context_analyzer.py  # Context analysis & optimization
│   ├── model.py             # Model interface
│   ├── auth.py              # API key authentication
│   ├── cache.py             # Response caching
│   ├── exceptions.py        # Custom exception hierarchy
│   ├── language_detector.py # Language detection
│   ├── logger.py            # Structured logging
│   ├── notifications.py     # Notification system
│   ├── rate_limiter.py      # Rate limiting
│   ├── security_scanner.py  # Input security scanning
│   ├── telemetry.py         # Usage telemetry
│   ├── utils.py             # Shared utilities
│   │
│   ├── llm/                 # LLM backend abstraction
│   │   ├── base.py          # Abstract provider interface
│   │   ├── config.py        # LLM-specific config
│   │   ├── router.py        # Dynamic provider routing
│   │   ├── ollama_provider.py
│   │   └── opencode_zen_provider.py
│   │
│   └── reasoning/           # Reasoning engine
│       ├── cot.py           # Chain-of-Thought
│       └── prompts.py       # Reasoning prompts
│
├── tools/                   # Tool implementations
│   ├── file_ops.py          # Read/write/edit/glob/grep
│   ├── web_search.py        # Web fetch & search
│   ├── git_ops.py           # Git operations
│   ├── code_analysis.py     # Code scan & review
│   ├── data_analysis.py     # CSV/JSON/Excel analysis
│   ├── documents.py         # PDF/DOCX/image handling
│   ├── voice.py             # TTS & speech recognition
│   ├── multi_agent.py       # Council & delegate agents
│   ├── scheduler.py         # Task scheduling
│   ├── docker_sandbox.py    # Docker code execution
│   ├── long_term_memory.py  # Persistent memory tools
│   └── self_improve.py      # Self-analysis & improvement
│
├── rag/                     # Retrieval-Augmented Generation
│   ├── retriever.py         # Hybrid search (semantic + BM25)
│   ├── vector_store.py      # Vector database
│   └── embedder.py          # Text embeddings
│
├── plugins/                 # Plugin system
│   ├── __init__.py
│   └── examples/
│       └── weather.py       # Example weather plugin
│
├── web/                     # Frontend
│   ├── index.html           # UI structure
│   ├── style.css            # Styling
│   └── app.js               # Frontend logic
│
├── tests/                   # Test suite (71 unit + 12 E2E)
│   ├── comprehensive_test.py
│   ├── unit_tests.py
│   ├── conftest.py
│   ├── test_agent_core.py
│   ├── test_file_ops.py
│   ├── test_web_endpoints.py
│   ├── test_rag_web.py
│   ├── test_multi_agent.py
│   ├── test_security_scanner.py
│   ├── test_code_analysis.py
│   ├── test_data_analysis.py
│   ├── test_scheduler.py
│   ├── test_long_term_memory.py
│   ├── test_notifications.py
│   ├── test_telemetry.py
│   ├── test_context_analyzer.py
│   ├── test_language_detector.py
│   ├── test_env_loader.py
│   ├── test_*.py
│   └── llm/
│       ├── test_base.py
│       ├── test_config.py
│       ├── test_router.py
│       ├── test_ollama_provider.py
│       └── test_opencode_zen_provider.py
│
├── benchmark_*.py           # Performance benchmarks
├── users.json               # User accounts & API keys
├── memory_store.json        # Persistent conversation store
├── install.ps1              # Windows installer
├── launcher.ps1             # Convenience launcher
└── requirements.txt         # Python dependencies
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- (Optional) [Ollama](https://ollama.ai) for local LLM
- (Optional) Docker for sandboxed code execution

### Installation

```powershell
# Clone & install
pip install -r requirements.txt

# Or use the automated installer
.\install.ps1
```

### Start the Server

```powershell
# Web interface mode
python web.py

# CLI mode
python main.py --cli

# Specify backend and model
python main.py --web --model "path/to/model.gguf"
```

### Open in Browser

Navigate to **http://127.0.0.1:8080**

---

## Configuration

### config.txt (user-editable)

```ini
# Backend: "ollama", "gpt4all", or "llama" (direct GGUF)
BACKEND = gpt4all

# Ollama
OLLAMA_MODEL = qwen2.5:7b
OLLAMA_BASE = http://127.0.0.1:11434

# GPT4All
GPT4ALL_MODEL = Phi-3-mini-4k-instruct.Q4_0.gguf
GPT4ALL_MODEL_DIR = C:\Users\coman\.cache\gpt4all

# Direct GGUF (llama-cpp-python)
MODEL_PATH =
N_GPU_LAYERS = -1
N_THREADS = 6

# Common
N_CTX = 32768
TEMP = 0.7
MAX_TOKENS = 8192
SYSTEM_PROMPT = You are a helpful, capable AI assistant...
WEB_HOST = 127.0.0.1
WEB_PORT = 8080
DB_PATH = memory_store.json

# Performance
FAST_MODE = auto
CACHE_TTL = 300
RAG_ENABLED = true

# Tools filter (empty = all enabled)
TOOLS_ENABLED =
```

### .env (secrets)

```
SECRET_KEY=change-me-in-production
API_KEY_HASH_SALT=change-me-in-production
CORS_ORIGINS=http://localhost:8080
LOG_LEVEL=INFO
```

---

## LLM Backends

The agent supports **three backends** with automatic fallback:

| Backend | Provider | Model Format | Requirements |
|---------|----------|-------------|-------------|
| `ollama` | Ollama | Any Ollama model (`qwen2.5:7b`, `llama3`, `deepseek`) | Ollama installed & running |
| `gpt4all` | GPT4All Python | `.gguf` downloaded by GPT4All | `gpt4all` package |
| `llama` | llama-cpp-python | Direct `.gguf` file path | `llama-cpp-python` |

The **LLM Router** (`core/llm/router.py`) selects the best available backend, with `auto` mode picking the first working provider.

---

## Web Interface

A modern, responsive single-page application with:

- **Streaming responses** via Server-Sent Events (SSE)
- **Tool call visualization** with expandable details
- **Conversation management** (new, list, delete)
- **File upload** with drag-and-drop
- **Dark/light theme**
- **Mobile-friendly** layout

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/status` | System status & health |
| `GET` | `/stats` | Tool usage statistics |
| `GET` | `/tools` | List all available tools |
| `POST` | `/chat` | Send message (SSE streaming) |
| `POST` | `/conversations/new` | Create new conversation |
| `GET` | `/conversations` | List all conversations |
| `GET` | `/conversations/{id}` | Get conversation history |
| `DELETE` | `/conversations/{id}` | Delete a conversation |
| `POST` | `/upload` | Upload a document |

---

## CLI Mode

Run in interactive CLI mode:

```powershell
python main.py --cli
```

Commands available in CLI:

| Command | Description |
|---------|-------------|
| `/new` | Start a new conversation |
| `/tools` | List all available tools |
| `/fast` | Toggle fast mode |
| `/rag` | Toggle RAG search |
| `/tools enable <name>` | Enable a specific tool |
| `/tools disable <name>` | Disable a specific tool |
| `/tools on` | Enable all tools |
| `/tools off` | Disable all tools |
| `/quit` | Exit |

---

## Tool Categories

| # | Category | Tools | Description |
|---|----------|-------|-------------|
| 1 | **Basic** | `datetime`, `calculator` | Date/time, math operations |
| 2 | **File** | `read`, `write`, `edit`, `glob`, `grep` | Full file system access |
| 3 | **Web** | `fetch`, `search`, `scrape` | Internet access & data retrieval |
| 4 | **Git** | `status`, `diff`, `log`, `commit`, `push` | Git operations |
| 5 | **Code** | `scan`, `review`, `refactor` | Code analysis & improvement |
| 6 | **Data** | `csv`, `json`, `excel` | Data processing & analysis |
| 7 | **Documents** | `pdf`, `docx`, `images` | Document parsing & generation |
| 8 | **Voice** | `listen`, `speak` | Speech input/output |
| 9 | **Multi-Agent** | `council`, `delegate` | Agent collaboration & delegation |
| 10 | **Scheduler** | `schedule`, `list`, `remove` | Recurring task scheduling |
| 11 | **Docker** | `run`, `images`, `exec` | Secure sandboxed execution |
| 12 | **Self-Improve** | `analyze`, `review`, `improve` | Self-enhancement & optimization |
| 13 | **Memory** | `recall`, `remember`, `summarize` | Long-term persistent memory |

---

## Plugin System

Extend the agent with custom tools:

```python
# plugins/my_plugin.py
from core.tools import register_tool

@register_tool(
    name="my_tool",
    description="Description of my tool",
    category="Custom"
)
def my_tool(param1: str, param2: int = 42) -> str:
    """Tool implementation"""
    return f"Result: {param1} x {param2}"
```

All plugins in the `plugins/` directory are auto-discovered on startup.

---

## RAG Engine

Hybrid retrieval combining **semantic vector search** and **BM25 keyword search**:

1. **Embedder** (`rag/embedder.py`): Converts text to vector embeddings (supports multiple backends)
2. **Vector Store** (`rag/vector_store.py`): Stores and searches embeddings
3. **Retriever** (`rag/retriever.py`): Hybrid search merging semantic + keyword results with configurable weights

Enable/disable via `RAG_ENABLED` in `config.txt` or `/rag` toggle in CLI.

---

## Security

| Feature | Description |
|---------|-------------|
| **Path Traversal Protection** | Prevents directory traversal attacks in file operations |
| **Code Execution Hardening** | Blocks dangerous Python patterns in sandbox |
| **Input Sanitization** | Validates and cleans all user input |
| **Rate Limiting** | Per-IP and per-user rate limits |
| **API Key Authentication** | HMAC-signed API keys with role-based access |
| **CORS Protection** | Dynamic CORS middleware with origin validation |
| **File Upload Limits** | 10MB max with allowed extension whitelist |
| **Context Window Limiting** | Prevents token overflow attacks |
| **Security Scanner** | Built-in regex-based threat detection |

---

## Testing

```powershell
# Run comprehensive test suite (71 tests)
python tests/comprehensive_test.py

# Run unit tests
python tests/unit_tests.py

# Run security-specific tests
python tests/final_check.py

# Run web endpoint tests
python -m pytest tests/test_web_endpoints.py -v

# Run all tests with coverage
python -m pytest tests/ --cov=core --cov=tools --cov=rag -v
```

Test categories: agent core, file ops, web endpoints, RAG, multi-agent, security scanner, code analysis, data analysis, scheduler, long-term memory, notifications, telemetry, context analyzer, language detector, env loader, and all LLM providers.

---

## Benchmarks

| Script | What it measures |
|--------|-----------------|
| `benchmark_tool_registry.py` | Tool registry caching performance (list_tools + format_for_prompt) |
| `benchmark_cors_compare.py` | Per-request vs keyed-cached CORS middleware comparison |
| `benchmark_cors_middleware.py` | Dynamic CORS middleware rebuild cost |

```powershell
python benchmark_tool_registry.py
python benchmark_cors_compare.py 3000
python benchmark_cors_middleware.py
```

---

## Improvement Log

See `IMPROVEMENT_LOG.md` for a detailed history of optimization rounds:

| Round | Focus | Key Changes |
|-------|-------|-------------|
| **R1** | Performance | Lazy loading, async I/O, RAG optimization, 28 new tests |
| **R2** | Code Quality | Deduplication, shared utilities, simplification |
| **R3** | Optimization | Removed redundant `memory.load()`, simplified system prompt builder |
| **R4** | Bug Fix | Fixed CORS module-level cache poisoning with `_DynamicCORSMiddleware` |
| **R5** | Performance | Keyed CORS cache yielding ~3% latency improvement (78 µs) |

**Final metrics**: 46 files, ~10,984 lines, 8,636 Python lines, 71 tests passing, 12 E2E tests.

---

## Project Structure

```
E:\AI\
├── main.py                     # CLI entry point
├── web.py                      # FastAPI server
├── config.py                   # Config loader
├── config.txt                  # User settings
├── requirements.txt            # Dependencies
├── install.ps1                 # Windows installer
├── launcher.ps1                # Convenience launcher
├── .env.example                # Environment template
├── users.json                  # User accounts
├── memory_store.json           # Conversation store
│
├── core/                       # Core engine (25 files)
├── tools/                      # Tool implementations (13 files)
├── rag/                        # RAG engine (4 files)
├── plugins/                    # Plugin system
├── web/                        # Frontend (HTML/CSS/JS)
├── config/                     # Config modules
├── models/                     # Model storage
├── tests/                      # Test suite (22+ files)
├── docs/                       # Documentation
└── logs/                       # Log output
```

---

## License

MIT
