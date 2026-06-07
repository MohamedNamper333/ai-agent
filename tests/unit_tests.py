"""Unit Tests for AI Agent"""
import sys
import os
import json
import time
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import Mock, patch, MagicMock


class TestToolResult:
    def test_tool_result_creation(self):
        from core.tools import ToolResult
        result = ToolResult(
            tool_name="test",
            success=True,
            result="output"
        )
        assert result.tool_name == "test"
        assert result.success is True
        assert result.result == "output"
        assert result.timestamp != ""

    def test_tool_result_failure(self):
        from core.tools import ToolResult
        result = ToolResult(
            tool_name="test",
            success=False,
            result="",
            error="Something went wrong"
        )
        assert result.success is False
        assert result.error == "Something went wrong"


class TestTool:
    def test_tool_creation(self):
        from core.tools import Tool
        tool = Tool(
            name="test_tool",
            description="A test tool",
            func=lambda: "result",
            category="test"
        )
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert tool.category == "test"

    def test_tool_execution(self):
        from core.tools import Tool
        tool = Tool(
            name="add",
            description="Add two numbers",
            func=lambda a, b: a + b,
            category="math"
        )
        result = tool.run(a=2, b=3)
        assert result.success is True
        assert result.result == "5"

    def test_tool_error_handling(self):
        from core.tools import Tool
        def failing_func():
            raise ValueError("Test error")
        
        tool = Tool(
            name="failing",
            description="A failing tool",
            func=failing_func,
            category="test"
        )
        result = tool.run()
        assert result.success is False
        assert "Test error" in result.error

    def test_tool_stats(self):
        from core.tools import Tool
        tool = Tool(
            name="test",
            description="Test",
            func=lambda: "ok"
        )
        tool.run()
        tool.run()
        stats = tool.get_stats()
        assert stats["calls"] == 2
        assert stats["name"] == "test"


class TestToolRegistry:
    def test_registry_creation(self):
        from core.tools import ToolRegistry
        registry = ToolRegistry()
        tools = registry.list_tools()
        assert len(tools) > 0

    def test_register_tool(self):
        from core.tools import ToolRegistry, Tool
        registry = ToolRegistry()
        initial_count = len(registry.list_tools())
        
        tool = Tool(
            name="custom_tool",
            description="Custom",
            func=lambda: "custom"
        )
        registry.register(tool)
        
        assert len(registry.list_tools()) == initial_count + 1
        assert registry.get("custom_tool") is not None

    def test_get_unknown_tool(self):
        from core.tools import ToolRegistry
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_list_tools_by_category(self):
        from core.tools import ToolRegistry
        registry = ToolRegistry()
        categories = registry.list_tools_by_category()
        assert len(categories) > 0
        assert "basic" in categories

    def test_format_for_prompt(self):
        from core.tools import ToolRegistry
        registry = ToolRegistry()
        prompt = registry.format_for_prompt()
        assert "Available Tools" in prompt
        assert "tool_calls" in prompt


class TestRateLimiter:
    def test_rate_limiter_creation(self):
        from core.rate_limiter import RateLimiter
        limiter = RateLimiter(max_requests=10, window_seconds=60)
        assert limiter.max_requests == 10

    def test_allows_requests_under_limit(self):
        from core.rate_limiter import RateLimiter
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        
        for _ in range(5):
            assert limiter.is_allowed("test_key") is True

    def test_blocks_requests_over_limit(self):
        from core.rate_limiter import RateLimiter
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        
        for _ in range(3):
            limiter.is_allowed("test_key")
        
        assert limiter.is_allowed("test_key") is False

    def test_get_remaining(self):
        from core.rate_limiter import RateLimiter
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        
        limiter.is_allowed("test_key")
        remaining = limiter.get_remaining("test_key")
        assert remaining == 4

    def test_tiered_rate_limiter(self):
        from core.rate_limiter import TieredRateLimiter
        limiter = TieredRateLimiter()
        
        assert limiter.is_allowed("user1", "basic") is True
        assert limiter.is_allowed("user1", "premium") is True


class TestAuthManager:
    def test_auth_manager_creation(self):
        from core.auth import AuthManager
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            manager = AuthManager(db_path=temp_path)
            manager.load()
            assert manager.get_user_count() == 0
        finally:
            os.unlink(temp_path)

    def test_create_user(self):
        from core.auth import AuthManager, UserRole
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            manager = AuthManager(db_path=temp_path)
            manager.load()
            
            user = manager.create_user("testuser", UserRole.BASIC)
            assert user.username == "testuser"
            assert user.role == UserRole.BASIC
            assert user.api_key.startswith("sk_")
            assert manager.get_user_count() == 1
        finally:
            os.unlink(temp_path)

    def test_get_user_by_api_key(self):
        from core.auth import AuthManager, UserRole
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            manager = AuthManager(db_path=temp_path)
            manager.load()
            
            user = manager.create_user("testuser", UserRole.BASIC)
            found = manager.get_user_by_api_key(user.api_key)
            assert found is not None
            assert found.user_id == user.user_id
        finally:
            os.unlink(temp_path)

    def test_invalid_api_key(self):
        from core.auth import AuthManager
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            manager = AuthManager(db_path=temp_path)
            manager.load()
            
            found = manager.get_user_by_api_key("invalid_key")
            assert found is None
        finally:
            os.unlink(temp_path)

    def test_create_session_token(self):
        from core.auth import AuthManager, UserRole
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            manager = AuthManager(db_path=temp_path)
            manager.load()
            
            user = manager.create_user("testuser", UserRole.BASIC)
            token = manager.create_session_token(user.user_id)
            assert token != ""
            
            found = manager.validate_session_token(token)
            assert found is not None
        finally:
            os.unlink(temp_path)


class TestSanitizeInput:
    def test_sanitize_empty(self):
        from web import sanitize_input
        assert sanitize_input("") == ""

    def test_sanitize_normal_text(self):
        from web import sanitize_input
        result = sanitize_input("Hello World")
        assert result == "Hello World"

    def test_sanitize_max_length(self):
        from web import sanitize_input
        result = sanitize_input("x" * 20000, max_length=10000)
        assert len(result) == 10000

    def test_sanitize_control_chars(self):
        from web import sanitize_input
        result = sanitize_input("test\x00\x08data")
        assert "\x00" not in result
        assert "\x08" not in result

    def test_sanitize_whitespace(self):
        from web import sanitize_input
        result = sanitize_input("  hello  ")
        assert result == "hello"


class TestValidateConversationId:
    def test_valid_id(self):
        from web import validate_conversation_id
        assert validate_conversation_id("conv_20260530_120000") is True

    def test_invalid_id(self):
        from web import validate_conversation_id
        assert validate_conversation_id("invalid_id") is False

    def test_empty_id(self):
        from web import validate_conversation_id
        assert validate_conversation_id("") is True


class TestContextManager:
    def test_build_prompt(self):
        from core.context import ContextManager
        manager = ContextManager()
        prompt = manager.build_prompt(
            user_input="Hello",
            tool_descriptions="Tools available"
        )
        assert "Hello" in prompt
        assert "Tools available" in prompt

    def test_build_with_tool_results(self):
        from core.context import ContextManager
        manager = ContextManager()
        results = [
            {"tool": "test", "result": "output", "success": True}
        ]
        prompt = manager.build_with_tool_results(
            user_input="Hello",
            tool_results=results
        )
        assert "Hello" in prompt
        assert "test" in prompt


class TestMemory:
    def test_memory_creation(self):
        from core.memory import ConversationMemory
        memory = ConversationMemory()
        assert memory.get_stats()["conversations"] == 0

    def test_add_message(self):
        from core.memory import ConversationMemory
        memory = ConversationMemory()
        memory.new_conversation()
        memory.add_message("user", "Hello")
        
        history = memory.get_history()
        assert len(history) == 1
        assert history[0]["role"] == "user"

    def test_multiple_messages(self):
        from core.memory import ConversationMemory
        memory = ConversationMemory()
        memory.new_conversation()
        memory.add_message("user", "Hello")
        memory.add_message("assistant", "Hi there!")
        
        history = memory.get_history()
        assert len(history) == 2

    def test_conversation_management(self):
        from core.memory import ConversationMemory
        memory = ConversationMemory()
        
        cid1 = memory.new_conversation("conv_test1")
        cid2 = memory.new_conversation("conv_test2")
        
        assert len(memory.list_conversations()) == 2
        
        memory.delete_conversation(cid1)
        assert len(memory.list_conversations()) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
