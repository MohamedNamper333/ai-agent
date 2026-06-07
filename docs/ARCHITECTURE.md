# AI Agent - System Architecture

## System Overview

AI Agent is a modular, tool-using AI assistant with planning capabilities, long-term memory, and a Retrieval-Augmented Generation (RAG) pipeline. It operates in two modes: CLI (interactive terminal) and Web (FastAPI server with streaming UI).

The core loop: User input → Context assembly → LLM planning/generation → Tool execution → Response synthesis → Memory storage.

## Core Components

### Agent (`core/agent.py`)

Orchestrator that ties all components together. Manages the chat lifecycle:

- **Planning mode**: For complex queries, generates an `ExecutionPlan` with `PlanStep`s, each containing `ToolCall`s. Executes steps sequentially, synthesizes results.
- **Fast mode**: For simple queries (detected via keyword analysis), skips planning and sends a direct prompt to the LLM.
- **Streaming mode**: Streams LLM output, detects tool calls in the response, executes them, and continues generation (up to `MAX_TOOL_LOOPS=10`).

Key classes:
- `ToolCall` — represents a single tool invocation with retry logic (max 3 attempts, exponential backoff).
- `PlanStep` — a step in an execution plan, containing one or more tool calls.
- `ExecutionPlan` — full plan with goal, steps, and aggregated statistics.

### Tools (`core/tools.py`)

`ToolRegistry` manages all tools with lazy-loading by category. Tools are registered as `Tool` instances wrapping callable functions.

**Tool categories and counts:**

| Category | Count | Description |
|----------|-------|-------------|
| `basic` | 3 | datetime, calculator |
| `file` | 9 | read/write/edit/glob/grep/list_dir/file_info/file_compare/batch_read |
| `web` | 3 | fetch_url, search_web, web_scrape |
| `git` | 8 | status/diff/log/branch/show/add/commit/blame |
| `code` | 9 | scan_project/review_code/analyze_imports/code_refactor/complexity_metrics/dependency_graph/analyze_security/analyze_code_quality/generate_test |
| `data` | 10 | analyze_csv/json/text/stats_summary/analyze_excel/sql_query/analyze_data_quality/correlation_analysis/generate_visualization/time_series_analysis |
| `documents` | 6 | read_pdf/read_docx/analyze_image/ocr_image/read_excel/html_to_text |
| `voice` | 3 | listen/speak/save_speech |
| `multi_agent` | 2 | council, delegate |
| `scheduler` | 3 | schedule_task/list_scheduled_tasks/remove_scheduled_task |
| `docker` | 2 | docker_run, docker_images |
| `self_improve` | 4 | self_analyze/self_review/suggest_improvements/apply_improvement |
| `memory` | 2 | recall, remember |

Security: `run_code` tool uses AST-based expression evaluation with restricted imports, sandboxed subprocess execution, and dangerous pattern blocking.

### Memory (`core/memory.py`)

`ConversationMemory` — JSON-file-backed conversation store with:
- Multiple conversations (keyed by `conv_YYYYMMDD_HHMMSS`)
- Token-aware trimming (default `max_tokens=6000`)
- Keyword search across all conversations
- Lazy save (every 5 messages or on explicit call)
- `format_for_llm()` produces `<|system|>`, `<|user|>`, `<|assistant|>` tagged format

Long-term memory (via `tools/long_term_memory.py`) provides cross-conversation recall and summary storage.

### Context (`core/context.py`)

`ContextManager` assembles prompts with proper token ordering:
1. System prompt
2. Tool descriptions
3. Conversation history
4. User input (+ optional tool results)
5. Assistant prefix

Supports three prompt variants: basic, with tool results, and planning.

### RAG (`rag/retriever.py`)

Hybrid search retriever combining:
- **Semantic search**: Vector embeddings via `Embedder`, stored in `VectorStore`
- **Keyword search**: BM25 scoring with IDF weighting
- **Score fusion**: Configurable alpha parameter (default 0.5) blending both scores
- **Semantic chunking**: Splits documents by paragraph/sentence boundaries (512 tokens, 64 overlap)
- **Query cache**: 5-minute TTL for repeated queries

Supports file/directory indexing, batch document insertion, and source-based deletion.

### Web Server (`web.py`)

FastAPI application with:
- CORS middleware (all origins allowed)
- SSE streaming for chat responses
- File upload for RAG document indexing
- User authentication (API key via Bearer token)
- Rate limiting per IP/tier
- Static file serving for web UI

## Data Flow

```
User Input
    │
    ▼
┌─────────────────────────────────────┐
│  Agent.chat()                       │
│  ├─ Memory.add_message("user", ...) │
│  ├─ RAG: Retriever.query_text()     │
│  ├─ Cache lookup (non-stream)       │
│  ├─ LTM recall                      │
│  └─ Check fast_mode                 │
└─────────────┬───────────────────────┘
              │
    ┌─────────┴─────────┐
    │                   │
    ▼                   ▼
┌──────────┐    ┌──────────────┐
│ Fast Mode│    │ Planning Mode│
│ Direct   │    │ _create_plan │
│ prompt   │    │ → ExecutionPlan
└────┬─────┘    └──────┬───────┘
     │                 │
     ▼                 ▼
┌──────────────┐  ┌────────────────┐
│ LLM.generate │  │ For each step: │
│ (stream)     │  │  ├─ Execute    │
└──────┬───────┘  │  │  tool_calls │
       │          │  ├─ Synthesize │
       │          │  │  step result│
       │          │  └─ Next step  │
       │          └───────┬────────┘
       │                  │
       ▼                  ▼
┌─────────────────────────────────────┐
│  Response synthesis                 │
│  ├─ Memory.add_message("assistant") │
│  ├─ Auto-summarize to LTM           │
│  ├─ Cache store                     │
│  └─ Return response                 │
└─────────────────────────────────────┘
```

## Security Model

1. **Code execution sandboxing**: `run_code` blocks dangerous imports (os, subprocess, sys, etc.), filters regex patterns, runs in isolated subprocess with empty env.
2. **Input sanitization**: Max 10,000 chars, control character stripping, conversation ID regex validation.
3. **File upload limits**: 10MB max, whitelist of allowed extensions (.txt, .md, .py, .json, .csv, .pdf, .docx).
4. **Rate limiting**: Per-IP rate limits with tier-based limits (anonymous < basic < admin).
5. **Authentication**: API key-based with role system (ADMIN, BASIC). Bearer token in HTTP headers.
6. **Tool enable/disable**: Tools can be individually or categorically disabled via config or runtime API.

## Technology Stack

| Layer | Technology |
|-------|-----------|
| LLM Backend | Ollama (qwen2.5:7b default) or llama.cpp (GGUF models) |
| Web Framework | FastAPI + Uvicorn |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector Store | Custom in-memory with JSON persistence |
| Authentication | FastAPI HTTPBearer + custom API key management |
| Memory | JSON file persistence |
| RAG | BM25 + cosine similarity hybrid search |
| Language | Python 3.10+ |

## Directory Structure

```
ai-agent/
├── main.py                 # Entry point (CLI + Web mode)
├── web.py                  # FastAPI server
├── config.py               # Configuration loader (config.txt)
├── config.txt              # Runtime configuration
├── core/
│   ├── agent.py            # Main Agent orchestrator
│   ├── tools.py            # Tool registry and execution
│   ├── memory.py           # Conversation memory
│   ├── context.py          # Prompt/context assembly
│   ├── model.py            # LLM backend abstraction
│   ├── cache.py            # Response caching
│   ├── auth.py             # Authentication & user management
│   ├── rate_limiter.py     # Rate limiting
│   └── logger.py           # Structured logging
├── tools/
│   ├── file_ops.py         # File operations
│   ├── web_search.py       # Web fetching & search
│   ├── git_ops.py          # Git operations
│   ├── code_analysis.py    # Code review & analysis
│   ├── data_analysis.py    # Data analysis & visualization
│   ├── documents.py        # PDF/DOCX/image processing
│   ├── voice.py            # Speech I/O
│   ├── multi_agent.py      # Multi-agent orchestration
│   ├── scheduler.py        # Task scheduling
│   ├── docker_sandbox.py   # Docker-based code execution
│   ├── self_improve.py     # Self-analysis & improvement
│   └── long_term_memory.py # Long-term memory store
├── rag/
│   ├── retriever.py        # Hybrid search retriever
│   ├── embedder.py         # Sentence embedding
│   └── vector_store.py     # Vector storage
├── plugins/
│   └── (plugin system)
├── web/
│   └── index.html          # Web UI frontend
├── models/                 # GGUF model files
└── docs/
    ├── ARCHITECTURE.md
    └── API_REFERENCE.md
```
