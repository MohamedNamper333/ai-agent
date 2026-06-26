"""Comprehensive tests for FastAPI web endpoints."""

import sys
import os
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client():
    """Return a fresh TestClient with clean ServerState for each test."""
    from web import app, ServerState
    app.state.srv = ServerState()
    client = TestClient(app, raise_server_exceptions=False)
    return client


def _patch_agent(agent_mock):
    """Patch the global ``agent`` object used by web.py."""
    return patch("web.agent", agent_mock)


def _patch_config(**overrides):
    """Patch ``config`` module attributes for a single test."""
    patches = []
    for attr, val in overrides.items():
        patches.append(patch.object(__import__("web", fromlist=["config"]).config if False else __import__("config"), attr, val))
    return patches


def _base_agent_mock():
    """Return a MagicMock shaped like the real ``agent`` global."""
    agent = MagicMock()
    agent.memory.conversations = {}
    agent.memory.current_id = ""
    agent.memory.list_conversations.return_value = []
    agent.memory.get_history.return_value = []
    agent.memory.new_conversation.return_value = "conv_20250101_120000"
    agent.memory.add_message.return_value = None
    agent.memory.format_for_llm.return_value = ""

    agent.tools.list_tools.return_value = []
    agent.tools.list_all_tools.return_value = []
    agent.tools.list_tools_by_category_all.return_value = {}
    agent.tools.get_registry_stats.return_value = {}
    agent.tools.get_enabled_count.return_value = 0
    agent.tools.is_enabled.return_value = False
    agent.tools.enable_tool.return_value = True
    agent.tools.disable_tool.return_value = True
    agent.tools.contains_tool_call.return_value = False
    agent.tools.format_for_prompt.return_value = ""
    agent.tools.get_tool_stats.return_value = {}
    agent.tools._tools = {}
    agent.tools.enable_category.return_value = 0
    agent.tools.disable_category.return_value = 0

    agent.plugins.plugins = {}
    agent.context.get_stats.return_value = {}
    agent.context.system_prompt = "system"
    agent.context.build_prompt.return_value = "prompt"

    agent._retriever = None
    agent._fast_mode = "auto"
    agent.get_execution_history.return_value = []

    agent.model = MagicMock()
    agent.achat = AsyncMock(return_value="hello")

    return agent


# ===========================================================================
# GET /status
# ===========================================================================

class TestGetStatus:
    """Tests for GET /status."""

    def test_status_returns_200(self):
        """Endpoint should return HTTP 200."""
        client = _make_client()
        agent = _base_agent_mock()
        with _patch_agent(agent):
            resp = client.get("/status")
        assert resp.status_code == 200

    def test_status_contains_required_keys(self):
        """Response must include model_loaded, model_name, conversations, current_conversation."""
        client = _make_client()
        agent = _base_agent_mock()
        with _patch_agent(agent):
            data = client.get("/status").json()
        assert "model_loaded" in data
        assert "model_name" in data
        assert "conversations" in data
        assert "current_conversation" in data

    def test_status_model_not_loaded(self):
        """When model is not loaded, model_name should be empty string."""
        client = _make_client()
        agent = _base_agent_mock()
        with patch("web.model_loaded", False), _patch_agent(agent):
            data = client.get("/status").json()
        assert data["model_loaded"] is False
        assert data["model_name"] == ""


# ===========================================================================
# GET /stats
# ===========================================================================

class TestGetStats:
    """Tests for GET /stats."""

    def test_stats_returns_200(self):
        client = _make_client()
        agent = _base_agent_mock()
        with _patch_agent(agent):
            resp = client.get("/stats")
        assert resp.status_code == 200

    def test_stats_contains_expected_keys(self):
        client = _make_client()
        agent = _base_agent_mock()
        with patch("web.model_loaded", False), _patch_agent(agent):
            data = client.get("/stats").json()
        for key in ("tool_count", "tool_count_total", "plugin_count", "model_loaded",
                     "tool_stats", "memory_stats", "cache_stats", "rag_stats",
                     "fast_mode", "rag_enabled"):
            assert key in data, f"Missing key: {key}"


# ===========================================================================
# GET /settings
# ===========================================================================

class TestGetSettings:
    """Tests for GET /settings."""

    def test_settings_returns_200(self):
        client = _make_client()
        agent = _base_agent_mock()
        with _patch_agent(agent):
            resp = client.get("/settings")
        assert resp.status_code == 200

    def test_settings_keys(self):
        client = _make_client()
        agent = _base_agent_mock()
        with _patch_agent(agent):
            data = client.get("/settings").json()
        for key in ("fast_mode", "rag_enabled", "cache_ttl", "model",
                     "tools_enabled", "tools_total", "cache_stats"):
            assert key in data, f"Missing key: {key}"


# ===========================================================================
# POST /settings/fast-mode
# ===========================================================================

class TestToggleFastMode:
    """Tests for POST /settings/fast-mode."""

    def test_toggle_returns_200(self):
        client = _make_client()
        agent = _base_agent_mock()
        with _patch_agent(agent):
            resp = client.post("/settings/fast-mode")
        assert resp.status_code == 200

    def test_toggle_returns_new_mode(self):
        client = _make_client()
        agent = _base_agent_mock()
        with patch("web.config") as mock_cfg, _patch_agent(agent):
            mock_cfg.FAST_MODE = "on"
            data = client.post("/settings/fast-mode").json()
        assert "fast_mode" in data
        assert data["fast_mode"] in ("on", "off", "auto")


# ===========================================================================
# POST /settings/rag
# ===========================================================================

class TestToggleRag:
    """Tests for POST /settings/rag."""

    def test_toggle_rag_returns_200(self):
        client = _make_client()
        agent = _base_agent_mock()
        with _patch_agent(agent):
            resp = client.post("/settings/rag")
        assert resp.status_code == 200

    def test_toggle_rag_flips_value(self):
        client = _make_client()
        agent = _base_agent_mock()
        with patch("web.config") as mock_cfg, _patch_agent(agent):
            mock_cfg.RAG_ENABLED = True
            data = client.post("/settings/rag").json()
        assert data["rag_enabled"] is False


# ===========================================================================
# GET /tools
# ===========================================================================

class TestListTools:
    """Tests for GET /tools."""

    def test_tools_returns_200(self):
        client = _make_client()
        agent = _base_agent_mock()
        with _patch_agent(agent):
            resp = client.get("/tools")
        assert resp.status_code == 200

    def test_tools_response_structure(self):
        client = _make_client()
        agent = _base_agent_mock()
        with _patch_agent(agent):
            data = client.get("/tools").json()
        assert "tools" in data
        assert "total" in data
        assert "enabled" in data

    def test_tools_category_dict(self):
        """tools value should be a dict mapping category -> list."""
        client = _make_client()
        agent = _base_agent_mock()
        tool_mock = MagicMock()
        tool_mock.name = "add"
        tool_mock.description = "add two numbers"
        agent.tools.list_tools_by_category_all.return_value = {
            "math": [tool_mock],
        }
        agent.tools.is_enabled.return_value = True
        with _patch_agent(agent):
            data = client.get("/tools").json()
        assert "math" in data["tools"]
        assert len(data["tools"]["math"]) == 1
        assert data["tools"]["math"][0]["name"] == "add"


# ===========================================================================
# POST /tools/{name}/enable
# ===========================================================================

class TestEnableTool:
    """Tests for POST /tools/{name}/enable."""

    def test_enable_tool_returns_200(self):
        client = _make_client()
        agent = _base_agent_mock()
        with _patch_agent(agent):
            resp = client.post("/tools/nonexistent/enable")
        assert resp.status_code == 200

    def test_enable_tool_success(self):
        client = _make_client()
        agent = _base_agent_mock()
        agent.tools.enable_tool.return_value = True
        with _patch_agent(agent):
            data = client.post("/tools/search/enable").json()
        assert data["status"] == "ok"
        assert data["name"] == "search"
        assert data["enabled"] is True

    def test_enable_tool_not_found(self):
        client = _make_client()
        agent = _base_agent_mock()
        agent.tools.enable_tool.return_value = False
        with _patch_agent(agent):
            data = client.post("/tools/ghost/enable").json()
        assert data["status"] == "not_found"


# ===========================================================================
# POST /tools/{name}/disable
# ===========================================================================

class TestDisableTool:
    """Tests for POST /tools/{name}/disable."""

    def test_disable_tool_returns_200(self):
        client = _make_client()
        agent = _base_agent_mock()
        with _patch_agent(agent):
            resp = client.post("/tools/search/disable")
        assert resp.status_code == 200

    def test_disable_tool_success(self):
        client = _make_client()
        agent = _base_agent_mock()
        agent.tools.disable_tool.return_value = True
        with _patch_agent(agent):
            data = client.post("/tools/search/disable").json()
        assert data["status"] == "ok"
        assert data["name"] == "search"
        assert data["enabled"] is False

    def test_disable_tool_not_found(self):
        client = _make_client()
        agent = _base_agent_mock()
        agent.tools.disable_tool.return_value = False
        with _patch_agent(agent):
            data = client.post("/tools/ghost/disable").json()
        assert data["status"] == "not_found"


# ===========================================================================
# GET /conversations
# ===========================================================================

class TestListConversations:
    """Tests for GET /conversations."""

    def test_conversations_returns_200(self):
        client = _make_client()
        agent = _base_agent_mock()
        with _patch_agent(agent):
            resp = client.get("/conversations")
        assert resp.status_code == 200

    def test_conversations_structure(self):
        client = _make_client()
        agent = _base_agent_mock()
        agent.memory.list_conversations.return_value = ["conv_a", "conv_b"]
        agent.memory.current_id = "conv_a"
        with _patch_agent(agent):
            data = client.get("/conversations").json()
        assert "conversations" in data
        assert "current" in data
        assert data["conversations"] == ["conv_a", "conv_b"]
        assert data["current"] == "conv_a"


# ===========================================================================
# POST /conversations/new
# ===========================================================================

class TestNewConversation:
    """Tests for POST /conversations/new."""

    def test_new_conversation_returns_200(self):
        client = _make_client()
        agent = _base_agent_mock()
        with _patch_agent(agent):
            resp = client.post("/conversations/new")
        assert resp.status_code == 200

    def test_new_conversation_returns_id(self):
        client = _make_client()
        agent = _base_agent_mock()
        agent.memory.new_conversation.return_value = "conv_20250101_120000"
        with _patch_agent(agent):
            data = client.post("/conversations/new").json()
        assert "conversation_id" in data
        assert data["conversation_id"] == "conv_20250101_120000"


# ===========================================================================
# GET /conversations/{conv_id}
# ===========================================================================

class TestGetConversation:
    """Tests for GET /conversations/{conv_id}."""

    def test_get_conversation_found(self):
        client = _make_client()
        agent = _base_agent_mock()
        agent.memory.conversations = {"conv_20250101_120000": []}
        agent.memory.get_history.return_value = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        with _patch_agent(agent):
            resp = client.get("/conversations/conv_20250101_120000")
        assert resp.status_code == 200
        data = resp.json()
        assert data["conversation_id"] == "conv_20250101_120000"
        assert len(data["messages"]) == 2

    def test_get_conversation_not_found(self):
        client = _make_client()
        agent = _base_agent_mock()
        agent.memory.conversations = {}
        with _patch_agent(agent):
            resp = client.get("/conversations/conv_99999999_999999")
        assert resp.status_code == 404


# ===========================================================================
# DELETE /conversations/{conv_id}
# ===========================================================================

class TestDeleteConversation:
    """Tests for DELETE /conversations/{conv_id}."""

    def test_delete_conversation_found(self):
        client = _make_client()
        agent = _base_agent_mock()
        agent.memory.conversations = {"conv_20250101_120000": []}
        with _patch_agent(agent):
            resp = client.delete("/conversations/conv_20250101_120000")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    def test_delete_conversation_not_found(self):
        client = _make_client()
        agent = _base_agent_mock()
        agent.memory.conversations = {}
        with _patch_agent(agent):
            resp = client.delete("/conversations/conv_99999999_999999")
        assert resp.status_code == 404


# ===========================================================================
# POST /chat
# ===========================================================================

class TestChat:
    """Tests for POST /chat."""

    def test_chat_missing_message_returns_422(self):
        """Empty or missing message should fail validation."""
        client = _make_client()
        agent = _base_agent_mock()
        with _patch_agent(agent):
            resp = client.post("/chat", json={"message": ""})
        assert resp.status_code == 422

    def test_chat_invalid_conversation_id_returns_422(self):
        """Invalid conversation_id format should fail validation."""
        client = _make_client()
        agent = _base_agent_mock()
        with _patch_agent(agent):
            resp = client.post("/chat", json={
                "message": "hello",
                "conversation_id": "bad-id",
            })
        assert resp.status_code == 422

    def test_chat_non_streaming_returns_text(self):
        """Non-streaming chat should return a JSON with 'text'."""
        client = _make_client()
        agent = _base_agent_mock()
        agent.achat = AsyncMock(return_value="Hi there!")
        with patch("web.model_loaded", True), _patch_agent(agent):
            resp = client.post("/chat", json={
                "message": "hello",
                "stream": False,
            })
        assert resp.status_code == 200
        data = resp.json()
        assert "text" in data
        assert data["text"] == "Hi there!"

    def test_chat_streaming_returns_sse(self):
        """Streaming chat should return event-stream content type."""
        client = _make_client()
        agent = _base_agent_mock()

        def _fake_generate(prompt, *args, **kwargs):
            yield "Hello "
            yield "world"

        agent.model.generate = MagicMock(side_effect=_fake_generate)

        with patch("web.model_loaded", True), _patch_agent(agent):
            resp = client.post("/chat", json={
                "message": "hello",
                "stream": True,
            })
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

    def test_chat_model_not_loaded_returns_503(self):
        """Chat should return 503 when model is not loaded and reload fails."""
        client = _make_client()
        agent = _base_agent_mock()
        with patch("web.model_loaded", False), \
             patch("web.model_name", ""), \
             patch("web.LLM", side_effect=Exception("no model")), \
             _patch_agent(agent):
            resp = client.post("/chat", json={
                "message": "hello",
                "stream": False,
            })
        assert resp.status_code == 503

    def test_chat_with_new_conversation(self):
        """Chat without conversation_id should create a new conversation."""
        client = _make_client()
        agent = _base_agent_mock()
        agent.memory.conversations = {}
        agent.memory.current_id = ""
        agent.achat = AsyncMock(return_value="ok")
        with patch("web.model_loaded", True), _patch_agent(agent):
            resp = client.post("/chat", json={
                "message": "start",
                "stream": False,
            })
        assert resp.status_code == 200
        agent.memory.new_conversation.assert_called()

    def test_chat_with_rag_enabled(self):
        """When use_rag=True and retriever exists, RAG query should be made."""
        client = _make_client()
        agent = _base_agent_mock()
        retriever_mock = MagicMock()
        retriever_mock.query_text.return_value = "RAG context"
        agent._retriever = retriever_mock
        agent.achat = AsyncMock(return_value="enriched answer")
        with patch("web.model_loaded", True), \
             patch("web.retriever", retriever_mock), \
             _patch_agent(agent):
            resp = client.post("/chat", json={
                "message": "question",
                "stream": False,
                "use_rag": True,
            })
        assert resp.status_code == 200
        retriever_mock.query_text.assert_called_once()

    def test_chat_streaming_tool_call_loop(self):
        """Streaming with a tool call in the response should execute the tool."""
        client = _make_client()
        agent = _base_agent_mock()

        call_count = 0

        def _fake_generate(prompt, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                yield 'I will use [tool:calculator]{"expression": "2+2"}'
            else:
                yield "The result is 4"

        agent.model.generate = MagicMock(side_effect=_fake_generate)
        agent.tools.contains_tool_call.side_effect = lambda t: "[tool:" in t
        agent.tools.parse_and_execute.return_value = (
            [{"tool": "calculator", "args": {"expression": "2+2"}}],
            [{"tool": "calculator", "result": "4", "success": True}],
        )
        agent.context.build_with_tool_results.return_value = "follow up"

        with patch("web.model_loaded", True), _patch_agent(agent):
            resp = client.post("/chat", json={
                "message": "calculate 2+2",
                "stream": True,
            })
        assert resp.status_code == 200
        body = resp.text
        assert "[DONE]" in body
        assert "tool_call" in body
        assert "tool_result" in body

    def test_chat_streaming_max_loops(self):
        """Streaming should stop after max tool-call loops."""
        client = _make_client()
        agent = _base_agent_mock()

        loop = 0

        def _fake_generate(prompt, *args, **kwargs):
            nonlocal loop
            loop += 1
            yield f'[tool:dummy]{{"x":{loop}}}'

        agent.model.generate = MagicMock(side_effect=_fake_generate)
        agent.tools.contains_tool_call.return_value = True
        agent.tools.parse_and_execute.return_value = (
            [{"tool": "dummy", "args": {}}],
            [{"tool": "dummy", "result": "ok", "success": True}],
        )
        agent.context.build_with_tool_results.return_value = "retry"

        with patch("web.model_loaded", True), _patch_agent(agent):
            resp = client.post("/chat", json={
                "message": "loop",
                "stream": True,
            })
        assert resp.status_code == 200
        body = resp.text
        assert "[DONE]" in body
        assert body.count("tool_call") == 5


class TestSwipeGesture:
    """Source-inspection tests for Task 8 — mobile swipe-to-toggle sidebar.

    The swipe gesture is implemented entirely in client-side JavaScript
    (no FastAPI endpoint), so we verify the implementation by reading
    the web/app.js and web/style.css source files and confirming that
    the required symbols and styles are present.
    """

    @staticmethod
    def _read_web(relative):
        from pathlib import Path
        return (
            Path(__file__).resolve().parent.parent.joinpath("web", relative)
            .read_text(encoding="utf-8", errors="replace")
        )

    def test_init_swipe_gesture_defined(self):
        src = self._read_web("app.js")
        assert re.search(r"function\s+initSwipeGesture\s*\(", src), (
            "initSwipeGesture() function is not defined in web/app.js — "
            "Task 8 (mobile swipe gesture) implementation is missing."
        )

    def test_init_swipe_gesture_registered_in_dom_content_loaded(self):
        src = self._read_web("app.js")
        assert "initSwipeGesture();" in src, (
            "initSwipeGesture() is not called in web/app.js — "
            "the function is defined but never invoked, so swipe gesture will not work."
        )

    def test_style_css_has_touch_action_manipulation(self):
        css = self._read_web("style.css")
        assert "touch-action" in css, (
            "web/style.css does not declare any touch-action rules — "
            "mobile tap targets may have 300ms double-tap-zoom delay."
        )
        assert re.search(r"touch-action\s*:\s*manipulation", css), (
            "web/style.css has touch-action but no `manipulation` value — "
            "double-tap zoom will not be suppressed on mobile."
        )


# ===========================================================================
# CORS — Round 4
# ===========================================================================

class TestCORS:
    """Tests for the CORS configuration resolver and installed middleware.

    The CORS policy lives in ``web._resolve_cors_config()``, which reads
    ``config.CORS_ORIGINS`` and ``config.WEB_PORT`` at call time. The
    middleware itself is installed once at import time, so the helper is
    tested directly via ``patch.object`` on the config module. One
    integration test verifies the actually-installed middleware handles a
    CORS preflight using the default origin.
    """

    @staticmethod
    def _resolve():
        from web import _resolve_cors_config
        return _resolve_cors_config()

    def test_explicit_list_enables_credentials(self):
        with patch.object(__import__("config"), "CORS_ORIGINS",
                          "http://example.com, http://other.com"):
            origins, creds = self._resolve()
        assert origins == ["http://example.com", "http://other.com"]
        assert creds is True

    def test_wildcard_disables_credentials(self):
        with patch.object(__import__("config"), "CORS_ORIGINS", "*"):
            origins, creds = self._resolve()
        assert origins == ["*"]
        assert creds is False

    def test_empty_string_falls_back_to_localhost(self):
        with patch.object(__import__("config"), "CORS_ORIGINS", ""), \
             patch.object(__import__("config"), "WEB_PORT", 8080):
            origins, creds = self._resolve()
        assert origins == ["http://localhost:8080"]
        assert creds is True

    def test_whitespace_only_falls_back_to_localhost(self):
        with patch.object(__import__("config"), "CORS_ORIGINS", "   "), \
             patch.object(__import__("config"), "WEB_PORT", 9000):
            origins, creds = self._resolve()
        assert origins == ["http://localhost:9000"]
        assert creds is True

    def test_none_value_falls_back_to_localhost(self):
        with patch.object(__import__("config"), "CORS_ORIGINS", None), \
             patch.object(__import__("config"), "WEB_PORT", 8080):
            origins, creds = self._resolve()
        assert origins == ["http://localhost:8080"]
        assert creds is True

    def test_comma_only_falls_back_to_localhost(self):
        with patch.object(__import__("config"), "CORS_ORIGINS", ",,,"), \
             patch.object(__import__("config"), "WEB_PORT", 8080):
            origins, creds = self._resolve()
        assert origins == ["http://localhost:8080"]
        assert creds is True

    def test_single_origin_no_trailing_whitespace(self):
        with patch.object(__import__("config"), "CORS_ORIGINS",
                          "  https://app.example.com  "):
            origins, creds = self._resolve()
        assert origins == ["https://app.example.com"]
        assert creds is True

    def test_middleware_preflight_with_default_origin(self):
        client = _make_client()
        agent = _base_agent_mock()
        with _patch_agent(agent):
            resp = client.options(
                "/chat",
                headers={
                    "Origin": "http://localhost:8080",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type",
                },
            )
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == \
            "http://localhost:8080"
        assert resp.headers.get("access-control-allow-credentials") == "true"
