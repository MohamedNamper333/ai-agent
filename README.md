# 🤖 AI Agent

**Open-source AI agent with 46 tools** — works 100% offline, zero API costs.

Built from scratch with Python, supports local LLMs via Ollama or direct GGUF files.

---

## 🚀 Quick Start

### 1. Install Ollama (recommended)

```powershell
# Download from https://ollama.com
# Then pull a model:
ollama pull qwen2.5:7b
```

### 2. Run the agent

```powershell
python main.py --cli     # Chat in terminal
python main.py --web     # Web interface at http://127.0.0.1:8080
```

---

## 🛠 46 Tools

| Category | Tools |
|----------|-------|
| **Basic** | `datetime`, `calculator`, `run_code`, `search_memory` |
| **Memory** | `recall`, `remember` — long-term memory across conversations |
| **File Ops** | `read_file`, `write_file`, `edit_file`, `glob`, `grep`, `list_dir`, `file_info` |
| **Web** | `fetch_url`, `search_web` — browse and search the internet |
| **Git** | `git_status`, `git_diff`, `git_log`, `git_branch`, `git_show` |
| **Code Analysis** | `scan_project`, `review_code`, `analyze_imports` |
| **Data Analysis** | `analyze_csv`, `analyze_json`, `analyze_text`, `stats_summary` |
| **Documents & Images** | `read_pdf`, `read_docx`, `analyze_image`, `ocr_image` |
| **Voice** | `listen`, `speak`, `save_speech` |
| **Multi-Agent** | `council` — convenes 3 specialist agents (Analyst, Programmer, Reviewer) |
| **Scheduling** | `schedule_task`, `list_scheduled_tasks`, `remove_scheduled_task` |
| **Docker Sandbox** | `docker_run` — execute code in isolated containers |
| **Self-Improvement** | `self_analyze`, `self_review`, `suggest_improvements`, `apply_improvement` |

### 🔌 Plugin System

Drop a `.py` file in `plugins/` — it auto-loads:

```python
# plugins/my_tool.py
from plugins import Plugin

class MyPlugin(Plugin):
    name = "my_tool"
    description = "Does something useful"

    def get_tools(self):
        return [{
            "name": "my_tool",
            "description": "What this tool does. Params: param1, param2",
            "func": self.run,
        }]

    def run(self, param1: str, param2: str = "") -> str:
        return f"Result: {param1} {param2}"
```

---

## 📁 Project Structure

```
ai-agent/
├── core/
│   ├── agent.py        # Agent loop (tool use, multi-step reasoning)
│   ├── model.py        # LLM backend (Ollama / llama-cpp)
│   ├── memory.py       # Conversation memory
│   ├── tools.py        # Tool registry (46 tools)
│   └── context.py      # Prompt building
├── tools/
│   ├── file_ops.py     # File read/write/edit/glob/grep
│   ├── web_search.py   # Internet search & fetch
│   ├── git_ops.py      # Git integration
│   ├── code_analysis.py# Code review & project scanning
│   ├── data_analysis.py# CSV/JSON/text analysis
│   ├── long_term_memory.py # Cross-session memory
│   ├── multi_agent.py  # Multi-agent orchestration
│   ├── documents.py    # PDF, DOCX, Image analysis
│   ├── voice.py        # Speech input/output
│   ├── scheduler.py    # Task scheduling
│   ├── docker_sandbox.py # Secure Docker execution
│   └── self_improve.py # Self-analysis & improvement
├── plugins/
│   ├── examples/
│       └── weather.py  # Example plugin
├── web/
│   ├── index.html      # Web UI
│   ├── style.css       # Dark theme
│   └── app.js          # Frontend logic
├── main.py             # CLI entry point
├── web.py              # FastAPI web server
├── config.py           # Configuration loader
├── install.ps1         # Setup script
└── launcher.ps1        # Quick launcher
```

---

## 🌐 Web Interface

```
python main.py --web
```

Opens at `http://127.0.0.1:8080` with:
- Chat interface with streaming responses
- RAG toggle (retrieve from uploaded documents)
- File upload for analysis
- Conversation history

---

## 💰 Cost

| Item | Cost |
|------|------|
| **Electricity** | ~$0.02-0.05/hour |
| **Ollama (local LLM)** | Free |
| **Python libraries** | Free |
| **APIs** | $0 — everything runs locally |
| **Total** | **~$3-5/month** (electricity only) |

---

## 🧠 Multi-Agent System

The `council` tool convenes 3 specialist agents:

1. **The Analyst** — data analysis, pattern finding, debugging
2. **The Programmer** — code writing, architecture, implementation
3. **The Reviewer** — code review, bug detection, security audit

A synthesis coordinator combines their responses into a unified answer.

---

## 🔧 Requirements

- **Python** 3.10+
- **Ollama** (recommended) or a GGUF model
- **GPU**: 4GB+ VRAM recommended (GTX 1650 works)
- **RAM**: 8GB+ (16GB+ recommended for 7B models)

---

## 📦 Optional Dependencies

```powershell
pip install PyPDF2        # PDF reading
pip install python-docx   # Word document reading
pip install Pillow        # Image analysis
pip install pytesseract   # OCR (requires Tesseract-OCR installed)
pip install SpeechRecognition  # Voice input
pip install pyttsx3       # Voice output (offline TTS)
pip install gtts          # Voice output (Google TTS)
```

---

## 🤝 Contributing

Add tools in `tools/` or create plugins in `plugins/`.  
All contributions welcome!

---

## 📄 License

MIT — free to use, modify, and distribute.
