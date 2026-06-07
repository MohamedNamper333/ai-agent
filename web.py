"""AI Agent - Web server (FastAPI)"""

import json
import os
import sys
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from core.model import LLM
from core.memory import ConversationMemory
from core.tools import ToolRegistry
from core.context import ContextManager
from core.agent import Agent
from rag.retriever import Retriever
import config


@dataclass
class ServerState:
    """Mutable server state — replaces module-level globals.

    All mutable state lives here and is attached to ``app.state`` so that
    endpoints access it via ``request.app.state.srv`` instead of reading
    module-level globals.  This makes the server testable without
    monkey-patching module attributes and is the standard FastAPI pattern
    for dependency injection.
    """

    agent: Agent = field(default_factory=Agent)
    model_loaded: bool = False
    model_name: str = ""
    retriever: Optional[Retriever] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    srv: ServerState = app.state.srv
    try:
        srv.agent.model = LLM(backend=config.BACKEND)
        srv.agent.model.load()
        srv.model_loaded = True
        if config.BACKEND == "ollama":
            srv.model_name = f"Ollama: {config.OLLAMA_MODEL}"
        else:
            srv.model_name = Path(config.MODEL_PATH).name
        srv.agent.memory.load()
        print(f"Model loaded: {srv.model_name}")
    except Exception as e:
        print(f"Model load (deferred): {e}")

    try:
        srv.retriever = Retriever()
        srv.retriever.load_or_init()
        print("RAG retriever initialized")
    except Exception as e:
        print(f"RAG init: {e}")

    yield


app = FastAPI(title="AI Agent", lifespan=lifespan)
app.state.srv = ServerState()


def _resolve_cors_config():
    raw = (config.CORS_ORIGINS or "").strip()
    if not raw:
        return [f"http://localhost:{config.WEB_PORT}"], True
    if raw == "*":
        return ["*"], False
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    if not origins:
        return [f"http://localhost:{config.WEB_PORT}"], True
    return origins, True


_cors_origins, _cors_allow_credentials = _resolve_cors_config()


class _DynamicCORSMiddleware:
    """CORS middleware that re-reads config per request with a small cache.

    The default Starlette ``CORSMiddleware`` freezes ``allow_origins`` and
    ``allow_credentials`` at install time (when ``app.add_middleware`` is
    called), so any runtime change to ``config.CORS_ORIGINS`` (by tests
    patching config, or by a future config-reload feature) would not be
    picked up by the running server. This wrapper delegates each HTTP
    request to a ``CORSMiddleware`` built from the current config, ensuring
    the active configuration is always honoured.

    To avoid allocating a fresh ``CORSMiddleware`` on every request, the
    last-built instance is cached and reused as long as the configuration
    key ``(tuple(origins), credentials)`` is unchanged. When the key
    changes (e.g. test patches ``config.CORS_ORIGINS``) the cache is
    invalidated and the next request rebuilds the middleware.
    """

    def __init__(self, app):
        self.app = app
        self._cache_key = None
        self._cached_cors = None

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        origins, credentials = _resolve_cors_config()
        key = (tuple(origins), credentials)
        if key != self._cache_key:
            self._cached_cors = CORSMiddleware(
                self.app,
                allow_origins=origins,
                allow_credentials=credentials,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            self._cache_key = key
        await self._cached_cors(scope, receive, send)


app.add_middleware(_DynamicCORSMiddleware)


class ChatRequest(BaseModel):
    conversation_id: str = Field(default="", pattern=r"^(|conv_[a-zA-Z0-9_]+)$")
    message: str = Field(..., min_length=1)
    stream: bool = True
    use_rag: bool = False


class ChatResponse(BaseModel):
    text: str


@app.get("/status")
async def get_status(request: Request):
    srv: ServerState = request.app.state.srv
    return {
        "model_loaded": srv.model_loaded,
        "model_name": srv.model_name if srv.model_loaded else "",
        "conversations": len(srv.agent.memory.conversations),
        "current_conversation": srv.agent.memory.current_id,
    }


@app.get("/stats")
async def get_stats(request: Request):
    srv: ServerState = request.app.state.srv
    return {
        "tool_count": len(srv.agent.tools.list_tools()),
        "tool_count_total": len(srv.agent.tools.list_all_tools()),
        "plugin_count": len(srv.agent.plugins.plugins) if srv.agent.plugins else 0,
        "model_loaded": srv.model_loaded,
        "tool_stats": getattr(srv.agent.tools, "get_tool_stats", lambda: {})(),
        "memory_stats": {
            "conversations": len(getattr(srv.agent.memory, "conversations", {})),
            "current_id": getattr(srv.agent.memory, "current_id", ""),
            "history": len(getattr(srv.agent.memory, "get_history", lambda: [])() or []),
        },
        "cache_stats": len(getattr(srv.agent.context, "_context_cache", {})),
        "rag_stats": {
            "enabled": bool(srv.retriever),
            "documents": len(getattr(srv.retriever, "docs", [])) if srv.retriever else 0,
        },
        "fast_mode": getattr(config, "FAST_MODE", "auto"),
        "rag_enabled": getattr(config, "RAG_ENABLED", False),
    }


@app.get("/settings")
async def get_settings(request: Request):
    srv: ServerState = request.app.state.srv
    return {
        "fast_mode": config.FAST_MODE,
        "rag_enabled": getattr(config, "RAG_ENABLED", False),
        "cache_ttl": config.CACHE_TTL,
        "model": srv.model_name,
        "tools_enabled": len(srv.agent.tools.list_tools()),
        "tools_total": len(srv.agent.tools.list_all_tools()),
        "cache_stats": len(getattr(srv.agent.context, "_context_cache", {})),
    }


@app.get("/tools")
async def list_tools_endpoint(request: Request):
    srv: ServerState = request.app.state.srv
    categories = srv.agent.tools.list_tools_by_category_all()
    tools_dict = {}
    for cat, tool_list in categories.items():
        tools_dict[cat] = [
            {"name": t.name, "description": t.description} for t in tool_list
        ]
    return {
        "tools": tools_dict,
        "total": len(srv.agent.tools.list_all_tools()),
        "enabled": len(srv.agent.tools.list_tools()),
    }


@app.post("/settings/fast-mode")
async def toggle_fast_mode():
    modes = ["on", "off", "auto"]
    try:
        current = config.FAST_MODE
        idx = modes.index(current) if current in modes else 0
    except (AttributeError, ValueError):
        idx = 0
    new_mode = modes[(idx + 1) % len(modes)]
    config.FAST_MODE = new_mode
    return {"fast_mode": new_mode}


@app.post("/settings/rag")
async def toggle_rag():
    current = bool(getattr(config, "RAG", False))
    new_val = not current
    config.RAG = new_val
    return {"rag_enabled": new_val}


@app.post("/tools/{name}/enable")
async def enable_tool_endpoint(name: str, request: Request):
    srv: ServerState = request.app.state.srv
    ok = srv.agent.tools.enable_tool(name)
    if ok:
        return {"status": "ok", "name": name, "enabled": True}
    return {"status": "not_found"}


@app.post("/tools/{name}/disable")
async def disable_tool_endpoint(name: str, request: Request):
    srv: ServerState = request.app.state.srv
    ok = srv.agent.tools.disable_tool(name)
    if ok:
        return {"status": "ok", "name": name, "enabled": False}
    return {"status": "not_found"}


@app.post("/conversations/new")
async def new_conversation(request: Request):
    srv: ServerState = request.app.state.srv
    cid = srv.agent.memory.new_conversation()
    return {"conversation_id": cid}


@app.get("/conversations")
async def list_conversations(request: Request):
    srv: ServerState = request.app.state.srv
    return {
        "conversations": srv.agent.memory.list_conversations(),
        "current": srv.agent.memory.current_id,
    }


@app.get("/conversations/{conv_id}")
async def get_conversation(conv_id: str, request: Request):
    srv: ServerState = request.app.state.srv
    if conv_id not in srv.agent.memory.conversations:
        raise HTTPException(404, "Conversation not found")
    msgs = srv.agent.memory.get_history(conv_id)
    return {"conversation_id": conv_id, "messages": msgs}


@app.delete("/conversations/{conv_id}")
async def delete_conversation(conv_id: str, request: Request):
    srv: ServerState = request.app.state.srv
    if conv_id in srv.agent.memory.conversations:
        srv.agent.memory.delete_conversation(conv_id)
        return {"status": "deleted"}
    raise HTTPException(404, "Conversation not found")


@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    srv: ServerState = request.app.state.srv
    if not srv.model_loaded:
        try:
            srv.agent.model = LLM(backend=config.BACKEND)
            srv.agent.model.load()
            srv.model_loaded = True
            srv.model_name = f"Ollama: {config.OLLAMA_MODEL}" if config.BACKEND == "ollama" else Path(config.MODEL_PATH).name
        except Exception as e:
            raise HTTPException(503, f"Model not loaded: {e}")

    cid = req.conversation_id or srv.agent.memory.current_id
    if cid and cid in srv.agent.memory.conversations:
        srv.agent.memory.current_id = cid
    else:
        cid = srv.agent.memory.new_conversation()

    if req.use_rag and srv.retriever:
        rag_context = srv.retriever.query_text(req.message)
        if rag_context:
            enriched = f"[Retrieved knowledge]\n{rag_context}\n\nUser question: {req.message}"
        else:
            enriched = req.message
    else:
        enriched = req.message

    # Multi-agent council route
    msg_lower = req.message.lower()
    if '/council' in msg_lower or 'council this' in msg_lower:
        topic = req.message.replace('/council', '').replace('council this', '').strip()
        from tools.multi_agent import MultiAgentOrchestrator
        council = MultiAgentOrchestrator(model=srv.agent.model)
        result = council.run_council(topic or enriched)
        srv.agent.memory.add_message("user", enriched)
        srv.agent.memory.add_message("assistant", result)
        return {"text": result, "council": True}

    if req.stream:
        async def event_stream():
            srv.agent.memory.add_message("user", enriched)
            history = srv.agent.memory.format_for_llm(
                srv.agent.context.system_prompt, include_system=False
            )
            tool_desc = srv.agent.tools.format_for_prompt()
            prompt = srv.agent.context.build_prompt(
                user_input=enriched,
                history=history,
                tool_descriptions=tool_desc,
            )

            full = ""
            for chunk in srv.agent.model.generate(prompt, stream=True):
                full += chunk
                yield f"data: {json.dumps({'text': chunk})}\n\n"

            max_loops = 5
            loop_count = 0
            while srv.agent.tools.contains_tool_call(full) and loop_count < max_loops:
                loop_count += 1
                tool_calls, tool_results = srv.agent.tools.parse_and_execute(full)
                for tc in tool_calls:
                    yield f"data: {json.dumps({'tool_call': tc})}\n\n"
                for tr in tool_results:
                    srv.agent.memory.add_message("system", f"Tool: {tr}")
                    yield f"data: {json.dumps({'tool_result': tr})}\n\n"

                prompt = srv.agent.context.build_with_tool_results(
                    user_input=enriched,
                    tool_results=tool_results,
                    history=history,
                    tool_descriptions=tool_desc,
                )
                full = ""
                for chunk in srv.agent.model.generate(prompt, stream=True):
                    full += chunk
                    yield f"data: {json.dumps({'text': chunk})}\n\n"

            srv.agent.memory.add_message("assistant", full)
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    else:
        response = await srv.agent.achat(enriched, stream=False)
        return {"text": response}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), request: Request = None):
    srv: ServerState = request.app.state.srv
    if srv.retriever is None:
        raise HTTPException(503, "RAG not initialized")

    content = await file.read()
    text = content.decode("utf-8", errors="replace")

    count = srv.retriever.add_document(text, {"source": file.filename})
    srv.retriever.save()
    return {"status": "ok", "file": file.filename, "chunks": count}


@app.get("/")
async def webui():
    return FileResponse(Path(__file__).parent / "web" / "index.html")


@app.get("/{path:path}")
async def static_files(path: str):
    filepath = Path(__file__).parent / "web" / path
    if filepath.exists() and filepath.is_file():
        return FileResponse(str(filepath))
    raise HTTPException(404)


def run_server(host: str = "", port: int = 0):
    host = host or config.WEB_HOST
    port = port or config.WEB_PORT
    print(f"Web UI: http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    run_server()
