# AI Agent — Personal Intelligence Platform

> Advanced AI agent with deductive reasoning, neural memory, deep code analysis, and agent swarms.
> Built on Qwen3:8b · OpenCodeZen · Ollama · GPT4All

![Tests](https://img.shields.io/badge/tests-821%20passing-brightgreen)
![Tools](https://img.shields.io/badge/tools-65%2B-blue)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)

---

## The 4 Pillars

| Pillar | File | Purpose |
|---|---|---|
| Deductive Engine | `core/reasoning/deductive_engine.py` | Tree-of-Thought: analyze → N plans → evaluate → decide → self-question |
| Neural Memory | `core/memory/neural_memory.py` | SQLite memory with Ebbinghaus decay + semantic retrieval |
| Obsidian Bridge | `core/memory/obsidian_bridge.py` | Permanent markdown knowledge graph (decisions/patterns/lessons) |
| Learning Engine | `core/learning_engine.py` | Captures interactions, tracks tool reliability, FAQ cache |

---

## Quick Start

```bash
git clone https://github.com/MohamedNamper333/ai-agent.git
cd ai-agent
pip install -r requirements.txt
cp .env.example .env      # Fill in your keys
ollama pull qwen3:8b
python main.py            # → http://localhost:8080
```

Docker:
```bash
docker compose up --build
```

---

## Architecture

```
core/
  reasoning/deductive_engine.py   Deductive thinking (ToT)
  memory/neural_memory.py         Conscious memory (SQLite)
  memory/obsidian_bridge.py       Subconscious (Markdown vault)
  llm/model_selector.py           Dynamic model switching
  llm/router.py                   Smart LLM routing
  learning_engine.py              Self-learning system
  agent.py                        Main agent (all pillars wired)
  auth.py                         JWT + API keys
  rate_limiter.py                 Tiered rate limiting

tools/
  agent_swarm.py                  8 specialist agents (3 patterns)
  code_optimizer.py               Code reduction up to 90%
  deep_analyzer.py                5-pass vulnerability finder
  web_search.py / documents.py / data_analysis.py / ...

rag/
  vector_store.py                 Numpy-cached (40x faster)
  embedder.py                     Ollama / sentence-transformers / TF-IDF
  retriever.py                    BM25 + semantic hybrid

Dockerfile                        Multi-stage production build
docker-compose.yml                App + Redis
```

---

## Model Selection

Switch models at runtime — no restart needed:

```bash
# List all available models
GET /models

# Switch model
POST /models/switch
{"model_id": "qwen3:8b", "provider": "ollama"}
{"model_id": "deepseek-v4-flash-free", "provider": "opencode_zen"}
{"model_id": "/path/to/model.gguf", "provider": "gpt4all"}
```

| Provider | Cost | Mode |
|---|---|---|
| Ollama (qwen3:8b default) | Free | Local |
| GPT4All (any .gguf) | Free | Local |
| OpenCodeZen (DeepSeek/Qwen/MiniMax+) | Free API | Cloud |

---

## Agent Swarm

```python
from tools.agent_swarm import AgentSwarm
swarm = AgentSwarm()

result = swarm.run_parallel("Review this architecture")   # fastest
result = swarm.run_pipeline("Build a payments API")       # most coherent
result = swarm.run_debate("GraphQL vs REST?")             # most accurate
result = swarm.run_auto("Analyze this codebase")          # auto-picks
```

**8 Specialists:** Analyst · Architect · Security · Optimizer · Critic · Researcher · Coder · Strategist

---

## Deep Analysis (5-Pass)

```python
from tools.deep_analyzer import DeepAnalyzer
result = DeepAnalyzer().analyze_file("mycode.py")
print(result.to_report())
# Pass 1: Surface (syntax, mutable defaults, TODOs)
# Pass 2: Security (16 patterns + CWE IDs)
# Pass 3: Logic (ZeroDivision, O(n^2), silent exceptions)
# Pass 4: Architecture (god classes, circular imports)
# Pass 5: LLM semantic (what static analysis misses)
```

---

## Auth

```bash
# One-time setup
curl -X POST http://localhost:8080/auth/init-admin
# Save the returned api_key

# Use in all requests
Authorization: Bearer sk_xxxx...
```

---

## API Endpoints

```
GET  /status              Server health (public)
GET  /models              All available models (public)
POST /models/switch       Switch active model (auth)
POST /auth/init-admin     Create first admin (one-time)
POST /chat                Send message (SSE streaming, auth)
GET  /pillars/status      4 pillars health (auth)
POST /pillars/think       Deep deductive reasoning (auth)
GET  /conversations       List conversations (auth)
GET  /tools               List all tools
POST /tools/{name}/enable Toggle tool (auth)
```

---

## Testing

```bash
python -m pytest tests/ -v
python -m pytest tests/test_fixes_verification.py -v   # 30 fix checks
python -m pytest tests/ --cov=core --cov-report=term
```

**Current:** 821 tests · 99.6% pass rate

---

## What Is Done

| Component | Status |
|---|---|
| Core Agent (all 4 pillars wired) | Done |
| DeductiveEngine (ToT + self-question) | Done |
| NeuralMemory (SQLite + decay) | Done |
| ObsidianBridge (markdown vault) | Done |
| LearningEngine (capture + FAQ) | Done |
| AgentSwarm (8 agents, 3 patterns) | Done |
| CodeOptimizer (rule + LLM passes) | Done |
| DeepAnalyzer (5-pass, CWE mapped) | Done |
| ModelSelector (Ollama+GPT4All+OCZ) | Done |
| Auth + Rate Limiting | Done |
| VectorStore (40x speedup) | Done |
| Docker + docker-compose | Done |
| RAG Hybrid (BM25 + semantic) | Done |

**Next Phase:** PostgreSQL · React UI · Monitoring · Billing

---

## License

MIT
