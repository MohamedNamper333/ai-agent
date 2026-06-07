# AI Agent - Advanced AI Assistant

## Overview
A powerful AI agent with 58+ tools, hybrid search, and a modern web interface.

## Features
- **58+ Tools** across 13 categories
- **Hybrid Search** (Semantic + BM25)
- **Streaming Chat** with tool visualization
- **Multi-Agent Council** for complex problems
- **Docker Sandbox** for secure code execution
- **Long-Term Memory** with automatic summarization

## Quick Start

### Prerequisites
- Python 3.11+
- Ollama (for local LLM)

### Installation
```bash
# Install Ollama
winget install Ollama.Ollama

# Pull model
ollama pull qwen2.5:7b

# Install dependencies
pip install -r requirements.txt

# Start server
python web.py
```

### Usage
Open http://127.0.0.1:8080 in your browser.

## Tool Categories

| Category | Tools | Description |
|----------|-------|-------------|
| Basic | datetime, calculator | Date/time and math operations |
| File | read, write, edit, glob, grep | File system operations |
| Web | fetch, search, scrape | Internet access |
| Git | status, diff, log, commit | Git operations |
| Code | scan, review, refactor | Code analysis |
| Data | CSV, JSON, Excel analysis | Data processing |
| Documents | PDF, DOCX, images | Document handling |
| Voice | listen, speak | Speech I/O |
| Multi-Agent | council, delegate | Agent collaboration |
| Scheduler | schedule, list, remove | Task scheduling |
| Docker | run, images | Secure code execution |
| Self-Improve | analyze, review, improve | Self-enhancement |
| Memory | recall, remember | Long-term memory |

## Security Features
- **Path Traversal Protection**: Prevents directory traversal attacks
- **Code Execution Hardening**: Blocks dangerous Python patterns
- **Input Sanitization**: Validates and cleans all user input
- **File Upload Limits**: Maximum 10MB with allowed extensions
- **Context Window Limiting**: Prevents token overflow

## Architecture

```
ai-agent/
├── core/           # Core agent logic
│   ├── agent.py    # Main agent loop
│   ├── tools.py    # Tool registry
│   ├── memory.py   # Conversation memory
│   └── context.py  # Context management
├── rag/            # Retrieval-Augmented Generation
│   ├── retriever.py    # Hybrid search
│   ├── vector_store.py # Vector database
│   └── embedder.py     # Text embeddings
├── tools/          # Tool implementations
│   ├── file_ops.py     # File operations
│   ├── web_search.py   # Web tools
│   ├── git_ops.py      # Git tools
│   └── ...             # 20+ tool modules
├── plugins/        # Plugin system
├── web/            # Web interface
│   ├── index.html  # UI structure
│   ├── style.css   # Styling
│   └── app.js      # Frontend logic
├── web.py          # FastAPI server
├── config.py       # Configuration
└── requirements.txt
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /status | System status |
| GET | /stats | Tool statistics |
| GET | /tools | List all tools |
| POST | /chat | Send message |
| POST | /conversations/new | New conversation |
| GET | /conversations | List conversations |
| GET | /conversations/{id} | Get conversation |
| DELETE | /conversations/{id} | Delete conversation |
| POST | /upload | Upload document |

## Configuration

Edit `config.py`:
```python
BACKEND = "ollama"          # or "llama_cpp"
OLLAMA_MODEL = "qwen2.5:7b"
WEB_HOST = "127.0.0.1"
WEB_PORT = 8080
```

## Testing

```bash
# Run all tests
python tests/comprehensive_test.py

# Run security tests only
python tests/final_check.py
```

## License
MIT
