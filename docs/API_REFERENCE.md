# AI Agent - API Reference

Base URL: `http://127.0.0.1:8080` (default)

## Authentication

API key authentication via `Authorization: Bearer <api_key>` header.

**Roles:** `admin`, `basic`

- **Unauthenticated**: Rate-limited, basic access
- **basic**: Standard rate limits
- **admin**: Full access, user management

## Rate Limiting

Per-IP rate limiting with tier-based limits. Returns `429` with `X-RateLimit-Remaining` header when exceeded.

## Endpoints

### System

#### `GET /status`

Returns server and model status.

**Response:**
```json
{
  "model_loaded": true,
  "model_name": "Ollama: qwen2.5:7b",
  "conversations": 3,
  "current_conversation": "conv_20260531_120000"
}
```

#### `GET /stats`

Returns detailed system statistics.

**Response:**
```json
{
  "tool_count": 42,
  "tool_count_total": 65,
  "plugin_count": 0,
  "model_loaded": true,
  "tool_stats": {
    "total_tools": 65,
    "total_calls": 150,
    "total_errors": 3,
    "error_rate": "2.0%",
    "categories": 13
  },
  "memory_stats": { "max_tokens": 8192, "system_prompt_length": 45, ... },
  "cache_stats": { "size": 12, "hits": 45, "misses": 20 },
  "rag_stats": { "vector_entries": 340, ... },
  "fast_mode": "auto",
  "rag_enabled": true
}
```

#### `GET /settings`

Returns current configuration.

**Response:**
```json
{
  "fast_mode": "auto",
  "rag_enabled": true,
  "cache_ttl": 300,
  "model": "qwen2.5:7b",
  "tools_enabled": 42,
  "tools_total": 65,
  "cache_stats": { ... }
}
```

#### `POST /settings/fast-mode`

Cycles fast mode: `on` → `off` → `auto` → `on`.

**Response:**
```json
{ "fast_mode": "auto" }
```

#### `POST /settings/rag`

Toggles RAG on/off.

**Response:**
```json
{ "rag_enabled": true }
```

### Chat

#### `POST /chat`

Send a message and receive a response (streaming or non-streaming).

**Request:**
```json
{
  "message": "Hello, how are you?",
  "conversation_id": "conv_20260531_120000",
  "stream": true,
  "use_rag": false
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `message` | string | yes | — | User message (max 10,000 chars) |
| `conversation_id` | string | no | current | Conversation ID (format: `conv_YYYYMMDD_HHMMSS`) |
| `stream` | bool | no | `true` | Enable SSE streaming |
| `use_rag` | bool | no | `false` | Enable RAG context retrieval |

**Non-streaming response:**
```json
{ "text": "I'm doing well, thanks!" }
```

**Streaming response (SSE):**
```
data: {"text": "I'm"}

data: {"text": " doing"}

data: {"text": " well"}

data: {"tool_call": {"name": "calculator", "arguments": {"expr": "2+2"}}}

data: {"tool_result": {"name": "calculator", "result": "4", "success": true}}

data: {"text": " The answer is 4."}

data: [DONE]
```

**Special commands in message:**
- `/council <topic>` or `council this <topic>` — triggers multi-agent council analysis

### Conversations

#### `POST /conversations/new`

Create a new conversation.

**Response:**
```json
{ "conversation_id": "conv_20260531_120000" }
```

#### `GET /conversations`

List all conversations.

**Response:**
```json
{
  "conversations": ["conv_20260531_120000", "conv_20260531_110000"],
  "current": "conv_20260531_120000"
}
```

#### `GET /conversations/{conv_id}`

Get messages for a conversation.

**Response:**
```json
{
  "conversation_id": "conv_20260531_120000",
  "messages": [
    { "role": "user", "content": "Hello", "timestamp": "2026-05-31T12:00:00" },
    { "role": "assistant", "content": "Hi there!", "timestamp": "2026-05-31T12:00:01" }
  ]
}
```

#### `DELETE /conversations/{conv_id}`

Delete a conversation.

**Response:**
```json
{ "status": "deleted" }
```

### Tools

#### `GET /tools`

List all tools grouped by category.

**Response:**
```json
{
  "tools": {
    "basic": [
      { "name": "datetime", "description": "Get current date and time", "enabled": true },
      { "name": "calculator", "description": "Evaluate a mathematical expression", "enabled": true }
    ],
    "file": [ ... ],
    "web": [ ... ]
  },
  "total": 65,
  "enabled": 42
}
```

#### `POST /tools/{name}/enable`

Enable a tool by name.

**Response:**
```json
{ "status": "ok", "name": "calculator", "enabled": true }
```

#### `POST /tools/{name}/disable`

Disable a tool by name.

**Response:**
```json
{ "status": "ok", "name": "calculator", "enabled": false }
```

#### `POST /tools/category/{category}/enable`

Enable all tools in a category.

**Response:**
```json
{ "status": "ok", "category": "file", "count": 9 }
```

#### `POST /tools/category/{category}/disable`

Disable all tools in a category.

**Response:**
```json
{ "status": "ok", "category": "file", "count": 9 }
```

#### `GET /tool-stats`

Per-tool usage statistics.

**Response:**
```json
{
  "stats": [
    {
      "name": "calculator",
      "calls": 45,
      "errors": 1,
      "error_rate": "2.2%",
      "avg_time": "0.003s",
      "total_time": "0.135s"
    },
    ...
  ]
}
```

### Execution History

#### `GET /execution-history`

Returns history of planning-mode execution plans.

**Response:**
```json
{
  "history": [
    {
      "goal": "Find all Python files and count lines",
      "status": "completed",
      "total_calls": 2,
      "successful": 2,
      "failed": 0,
      "steps": 3
    }
  ]
}
```

### Authentication

#### `POST /auth/register`

Register a new user.

**Query params:**
| Param | Type | Required | Default |
|-------|------|----------|---------|
| `username` | string | yes | — |
| `role` | string | no | `basic` |

**Response:**
```json
{
  "user_id": "usr_abc123",
  "username": "john",
  "api_key": "ak_...",
  "role": "basic"
}
```

#### `GET /auth/users`

List all users (admin only).

**Response:**
```json
{
  "users": [
    { "user_id": "usr_abc123", "username": "admin", "role": "admin", "api_key": "ak_..." },
    { "user_id": "usr_def456", "username": "john", "role": "basic", "api_key": "ak_..." }
  ]
}
```

### File Upload

#### `POST /upload`

Upload a document for RAG indexing.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | yes | File to upload (max 10MB) |

**Allowed extensions:** `.txt`, `.md`, `.py`, `.json`, `.csv`, `.pdf`, `.docx`

**Response:**
```json
{ "status": "ok", "file": "document.pdf", "chunks": 15 }
```

### Static Files

#### `GET /`

Serves the web UI (`web/index.html`).

#### `GET /{path}`

Serves static files from `web/` directory.

## Error Responses

| Status | Meaning |
|--------|---------|
| `400` | Bad request (invalid input, duplicate username) |
| `401` | Unauthorized (missing/invalid API key) |
| `403` | Forbidden (insufficient permissions) |
| `404` | Not found (conversation, file, tool) |
| `413` | File too large |
| `429` | Rate limit exceeded |
| `503` | Service unavailable (model not loaded, RAG not initialized) |

**Error format:**
```json
{ "detail": "Error message" }
```

## Configuration

Runtime configuration via `config.txt`:

| Key | Default | Description |
|-----|---------|-------------|
| `BACKEND` | `auto` | LLM backend (`auto`, `ollama`, `llama`) |
| `OLLAMA_MODEL` | `qwen2.5:7b` | Ollama model name |
| `OLLAMA_BASE` | `http://127.0.0.1:11434` | Ollama server URL |
| `MODEL_PATH` | (auto-detect) | Path to GGUF model file |
| `N_CTX` | `8192` | Context window size |
| `TEMP` | `0.7` | Sampling temperature |
| `MAX_TOKENS` | `2048` | Max generation tokens |
| `WEB_HOST` | `127.0.0.1` | Server bind host |
| `WEB_PORT` | `8080` | Server port |
| `FAST_MODE` | `auto` | Planning mode (`on`, `off`, `auto`) |
| `CACHE_TTL` | `300` | Response cache TTL (seconds) |
| `RAG_ENABLED` | `true` | Enable/disable RAG |
| `TOOLS_ENABLED` | (all) | Comma-separated tool filter |
