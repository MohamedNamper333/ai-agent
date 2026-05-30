"""AI Agent - Web server (FastAPI)"""

import json
import os
import sys
import uuid
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from core.model import LLM
from core.memory import ConversationMemory
from core.tools import ToolRegistry
from core.context import ContextManager
from core.agent import Agent
from rag.retriever import Retriever
import config

app = FastAPI(title="AI Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = Agent()
retriever: Optional[Retriever] = None
model_loaded = False
model_name = ""


class ChatRequest(BaseModel):
    conversation_id: str = ""
    message: str
    stream: bool = True
    use_rag: bool = False


class ChatResponse(BaseModel):
    text: str


@app.on_event("startup")
async def startup():
    global model_loaded, model_name, retriever
    try:
        agent.model = LLM(backend=config.BACKEND)
        agent.model.load()
        model_loaded = True
        if config.BACKEND == "ollama":
            model_name = f"Ollama: {config.OLLAMA_MODEL}"
        else:
            model_name = Path(config.MODEL_PATH).name
        agent.memory.load()
        print(f"Model loaded: {model_name}")
    except Exception as e:
        print(f"Model load (deferred): {e}")

    try:
        retriever = Retriever()
        retriever.load_or_init()
        print("RAG retriever initialized")
    except Exception as e:
        print(f"RAG init: {e}")


@app.get("/status")
async def get_status():
    return {
        "model_loaded": model_loaded,
        "model_name": model_name if model_loaded else "",
        "conversations": len(agent.memory.conversations),
    }


@app.get("/stats")
async def get_stats():
    return {
        "tool_count": len(agent.tools.list_tools()),
        "plugin_count": len(agent.plugins.plugins) if agent.plugins else 0,
        "model_loaded": model_loaded,
    }


@app.post("/conversations/new")
async def new_conversation():
    cid = agent.memory.new_conversation()
    return {"conversation_id": cid}


@app.get("/conversations")
async def list_conversations():
    return {
        "conversations": agent.memory.list_conversations(),
        "current": agent.memory.current_id,
    }


@app.get("/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    if conv_id not in agent.memory.conversations:
        raise HTTPException(404, "Conversation not found")
    msgs = agent.memory.get_history(conv_id)
    return {"conversation_id": conv_id, "messages": msgs}


@app.delete("/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    if conv_id in agent.memory.conversations:
        agent.memory.delete_conversation(conv_id)
        return {"status": "deleted"}
    raise HTTPException(404, "Conversation not found")


@app.post("/chat")
async def chat(req: ChatRequest):
    global model_loaded, model_name
    if not model_loaded:
        try:
            agent.model = LLM(backend=config.BACKEND)
            agent.model.load()
            model_loaded = True
            model_name = f"Ollama: {config.OLLAMA_MODEL}" if config.BACKEND == "ollama" else Path(config.MODEL_PATH).name
        except Exception as e:
            raise HTTPException(503, f"Model not loaded: {e}")

    cid = req.conversation_id or agent.memory.current_id
    if cid and cid in agent.memory.conversations:
        agent.memory.current_id = cid
    else:
        cid = agent.memory.new_conversation()

    if req.use_rag and retriever:
        rag_context = retriever.query_text(req.message)
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
        council = MultiAgentOrchestrator(model=agent.model)
        result = council.run_council(topic or enriched)
        agent.memory.add_message("user", enriched)
        agent.memory.add_message("assistant", result)
        return {"text": result, "council": True}

    if req.stream:
        async def event_stream():
            agent.memory.add_message("user", enriched)
            history = agent.memory.format_for_llm(
                agent.context.system_prompt, include_system=False
            )
            tool_desc = agent.tools.format_for_prompt()
            prompt = agent.context.build_prompt(
                user_input=enriched,
                history=history,
                tool_descriptions=tool_desc,
            )

            full = ""
            for chunk in agent.model.generate(prompt, stream=True):
                full += chunk
                yield f"data: {json.dumps({'text': chunk})}\n\n"

            max_loops = 3
            loop_count = 0
            while agent.tools.contains_tool_call(full) and loop_count < max_loops:
                loop_count += 1
                tool_results = agent.tools.parse_and_execute(full)
                for tr in tool_results:
                    agent.memory.add_message("system", f"Tool: {tr}")

                prompt = agent.context.build_with_tool_results(
                    user_input=enriched,
                    tool_results=tool_results,
                    history=history,
                    tool_descriptions=tool_desc,
                )
                full = ""
                for chunk in agent.model.generate(prompt, stream=True):
                    full += chunk
                    yield f"data: {json.dumps({'text': chunk})}\n\n"

            agent.memory.add_message("assistant", full)
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")
    else:
        response = agent.chat(enriched, stream=False)
        return {"text": response}


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if retriever is None:
        raise HTTPException(503, "RAG not initialized")

    content = await file.read()
    text = content.decode("utf-8", errors="replace")

    count = retriever.add_document(text, {"source": file.filename})
    retriever.save()
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
