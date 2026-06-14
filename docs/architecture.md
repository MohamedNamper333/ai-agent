# Architecture Overview

## System Design

The agent is built as a layered system:

```
┌─────────────────────────────────────────────┐
│              Web Interface                   │
│         FastAPI + SSE streaming              │
├─────────────────────────────────────────────┤
│               Core Agent                     │
│  ┌──────────┐ ┌──────────┐ ┌─────────────┐  │
│  │Deductive │ │  Neural  │ │  Learning   │  │
│  │ Engine   │ │  Memory  │ │   Engine    │  │
│  └──────────┘ └──────────┘ └─────────────┘  │
├─────────────────────────────────────────────┤
│              LLM Router                      │
│  Ollama ←→ OpenCodeZen ←→ GPT4All           │
├─────────────────────────────────────────────┤
│               Tools (65+)                    │
│  AgentSwarm · CodeOptimizer · DeepAnalyzer  │
│  WebSearch · Documents · DataAnalysis · ...  │
├─────────────────────────────────────────────┤
│         RAG (Hybrid BM25 + Semantic)         │
│  VectorStore (numpy cache) · Embedder        │
└─────────────────────────────────────────────┘
```

## Reasoning Flow

```
User Input
    │
    ▼
Neural Memory Recall (relevant past decisions)
    │
    ▼
FAQ Cache Check (repeated questions → instant answer)
    │
    ▼
RAG Context Enrichment
    │
    ▼
LLM Router (routes to best model by complexity)
    │
    ▼
Tool Execution (if needed, parallel with AgentSwarm)
    │
    ▼
Response + Learning Capture
    │
    ▼
Obsidian / NeuralMemory storage
```

## Deductive Engine (Tree-of-Thought)

```
Problem
  ├── Step 1: ANALYZE (root cause, constraints, unknowns)
  ├── Step 2: GENERATE (N candidate plans as structured JSON)
  ├── Step 3: EVALUATE (feasibility/risk/scalability/innovation scores)
  ├── Step 4: DECIDE (weighted composite + rejection explanation)
  └── Step 5: SELF-QUESTION ("Is there a better approach?")
```

## Memory Architecture

```
Conscious (fast, SQLite):
  MemoryNode {decision, reasoning, factors, outcome, importance}
  Ebbinghaus decay: importance × exp(-age/168h)
  Reinforcement: +0.05 importance per access

Subconscious (permanent, Obsidian vault):
  _decisions/   YYYY-MM-DD_slug.md  (wikilinked)
  _patterns/    learned_pattern.md
  _lessons/     lesson_TIMESTAMP.md
  _daily/       YYYY-MM-DD.md
```

## Agent Swarm Patterns

```
PARALLEL:  A ─┐
           B ─┼─► Synthesize ─► Result    (fastest)
           C ─┘

PIPELINE:  A ──► B ──► C ──► Synthesize  (most coherent)

DEBATE:    Round 1: A,B,C argue in parallel
           Round 2: A,B,C rebut in parallel
           Synthesize all ──► Result       (most accurate)
```

## Security Layers

1. Rate limiting: anonymous(20/min) · basic(60/min) · admin(1000/min)
2. Bearer token on all write endpoints
3. CORS: dynamic, cached, configurable
4. Code execution: subprocess isolation, empty environment
5. Upload: extension whitelist + 10MB limit
6. Path traversal: validate_path() on all file operations
