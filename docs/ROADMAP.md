# Development Roadmap

## Phase 1 — Infrastructure (Weeks 1-8) [IN PROGRESS]

**Goal:** Production-ready foundation

| Task | Status |
|---|---|
| Fix OllamaProvider model_ref bug | Done |
| requirements.txt complete | Done |
| Auth wired to all endpoints | Done |
| Rate limiting active | Done |
| VectorStore O(n) → cache | Done |
| Embedder fallback (TF-IDF) | Done |
| Docker + docker-compose | Done |
| DeductiveEngine | Done |
| NeuralMemory + Obsidian | Done |
| AgentSwarm | Done |
| CodeOptimizer | Done |
| DeepAnalyzer | Done |
| ModelSelector | Done |
| **SQLite → PostgreSQL** | Pending |
| **React.js frontend** | Pending |
| **Artifacts system** | Pending |

## Phase 2 — Intelligence (Weeks 9-18)

- ReAct reasoning loop (Reason + Act + Observe)
- Projects system (persistent context per project)
- Tool success rate → 95%+ (learning-based routing)
- Tavily/SerpAPI web search upgrade
- Multi-modal input (images via LLM vision)

## Phase 3 — Excellence (Weeks 19-28)

- React.js frontend with Artifacts panel
- Constitutional AI prompt engineering
- Browser automation (Playwright)
- OpenTelemetry + Prometheus + Grafana

## Phase 4 — Production (Weeks 29-36)

- Redis caching + connection pooling
- Load testing (k6), p95 < 2s
- OWASP security audit
- Stripe billing + usage tracking
- Public developer API

## Realistic KPIs

| Metric | Target | Note |
|---|---|---|
| Test coverage | >95% | Realistic |
| Tool success rate | >90% | Realistic |
| Response latency p95 | <3s | With Ollama local |
| Auth security | Zero public exploits | Realistic |
| Memory accuracy | >80% semantic recall | Realistic |
| ~~Memory < 2GB~~ | ~~Impossible~~ | Qwen3:8b needs 8-16GB |
| ~~1000 req/sec~~ | ~~Impossible~~ | Without $50K+ infra |
