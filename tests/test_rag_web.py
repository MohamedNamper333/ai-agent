"""Tests for RAG pipeline, Web API, and other uncovered areas."""
import sys
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestVectorStore:
    def test_vector_store_add_and_search(self):
        from rag.vector_store import VectorStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(db_path=os.path.join(tmpdir, "test.json"))
            store.add("Hello world", [1.0, 0.0, 0.0], {"source": "test"})
            store.add("Goodbye world", [0.0, 1.0, 0.0], {"source": "test"})
            store.add("Hello there", [0.9, 0.1, 0.0], {"source": "test2"})
            results = store.search([1.0, 0.0, 0.0], top_k=2)
            assert len(results) == 2
            assert results[0]["text"] == "Hello world"
            assert results[0]["score"] > 0.99

    def test_vector_store_delete(self):
        from rag.vector_store import VectorStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(db_path=os.path.join(tmpdir, "test.json"))
            store.add("Test", [1.0, 0.0])
            assert store.delete(0)
            assert len(store.entries) == 0
            assert not store.delete(0)

    def test_vector_store_delete_by_source(self):
        from rag.vector_store import VectorStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(db_path=os.path.join(tmpdir, "test.json"))
            store.add("Test1", [1.0, 0.0], {"source": "a"})
            store.add("Test2", [0.0, 1.0], {"source": "b"})
            store.add("Test3", [1.0, 0.0], {"source": "a"})
            deleted = store.delete_by_source("a")
            assert deleted == 2
            assert len(store.entries) == 1

    def test_vector_store_save_load(self):
        from rag.vector_store import VectorStore
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.json")
            store = VectorStore(db_path=path)
            store.add("Test", [1.0, 0.0], {"source": "test"})
            store.save()
            store2 = VectorStore(db_path=path)
            store2.load()
            assert len(store2.entries) == 1
            assert store2.entries[0]["text"] == "Test"

    def test_vector_store_empty_search(self):
        from rag.vector_store import VectorStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(db_path=os.path.join(tmpdir, "test.json"))
            results = store.search([1.0, 0.0], top_k=5)
            assert len(results) == 0

    def test_vector_store_stats(self):
        from rag.vector_store import VectorStore
        with tempfile.TemporaryDirectory() as tmpdir:
            store = VectorStore(db_path=os.path.join(tmpdir, "test.json"))
            store.add("Test1", [1.0, 0.0], {"source": "a"})
            store.add("Test2", [0.0, 1.0], {"source": "b"})
            stats = store.get_stats()
            assert stats["total_entries"] == 2
            assert stats["unique_sources"] == 2


class TestRetriever:
    def test_retriever_init(self):
        from rag.retriever import Retriever
        retriever = Retriever()
        assert retriever.embedder is not None
        assert retriever.store is not None
        assert retriever.doc_count == 0

    def test_retriever_chunk_text(self):
        from rag.retriever import Retriever
        retriever = Retriever()
        text = ("This is a test paragraph with enough words to exceed the chunk size limit. " * 10)
        chunks = retriever._chunk_text_semantic(text, chunk_size=200, min_chunk=20)
        assert len(chunks) >= 2
        assert all(isinstance(c, str) for c in chunks)

    def test_retriever_tokenization(self):
        from rag.retriever import Retriever
        retriever = Retriever()
        tokens = retriever._tokenize("Hello world, this is a test!")
        assert "hello" in tokens
        assert "world" in tokens
        assert "test" in tokens

    def test_retriever_bm25_score(self):
        from rag.retriever import Retriever
        retriever = Retriever()
        retriever.doc_count = 2
        retriever.doc_lengths = [5, 5]
        retriever.idf = {"hello": 1.0, "world": 1.0}
        retriever.corpus_tokens = [["hello", "world", "test"], ["hello", "foo", "bar"]]
        score = retriever._bm25_score(["hello"], retriever.corpus_tokens[0])
        assert score > 0


class TestCache:
    def test_cache_basic(self):
        from core.cache import CacheManager
        manager = CacheManager()
        cache = manager.get_cache("test", max_size=100, ttl=300)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None

    def test_cache_ttl(self):
        from core.cache import CacheManager
        manager = CacheManager()
        cache = manager.get_cache("test_ttl", max_size=100, ttl=1)
        cache.set("key1", "value1")
        import time
        time.sleep(1.1)
        assert cache.get("key1") is None

    def test_cache_max_size(self):
        from core.cache import CacheManager
        manager = CacheManager()
        cache = manager.get_cache("test_size", max_size=2, ttl=300)
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.set("k3", "v3")
        assert cache.get("k1") is None
        assert cache.get("k2") == "v2"
        assert cache.get("k3") == "v3"


class TestMemoryOptimization:
    def test_memory_dirty_flag(self):
        from core.memory import ConversationMemory
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "memory.json")
            mem = ConversationMemory(db_path=path)
            mem.new_conversation("test1")
            assert not mem._dirty
            mem.add_message("user", "Hello")
            assert mem._dirty or mem._last_save_count > 0

    def test_memory_save_if_dirty(self):
        from core.memory import ConversationMemory
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "memory.json")
            mem = ConversationMemory(db_path=path)
            mem.new_conversation("test1")
            mem.add_message("user", "Hello")
            mem.save_if_dirty()
            assert not mem._dirty
            assert os.path.exists(path)

    def test_memory_batch_save(self):
        from core.memory import ConversationMemory
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "memory.json")
            mem = ConversationMemory(db_path=path)
            mem.new_conversation("test1")
            for i in range(10):
                mem.add_message("user", f"Message {i}")
            assert mem._last_save_count == 10


class TestToolRegistryLazyLoading:
    def test_lazy_loading_basic(self):
        from core.tools import ToolRegistry
        registry = ToolRegistry()
        assert "basic" in registry._loaded_categories
        assert len(registry._tools) == 4

    def test_lazy_loading_ensure_all(self):
        from core.tools import ToolRegistry
        registry = ToolRegistry()
        registry._ensure_all()
        assert len(registry._tools) >= 60
        assert len(registry._loaded_categories) >= 10

    def test_lazy_loading_get_tool(self):
        from core.tools import ToolRegistry
        registry = ToolRegistry()
        tool = registry.get("read_file")
        assert tool is not None
        assert tool.name == "read_file"

    def test_lazy_loading_list_tools(self):
        from core.tools import ToolRegistry
        registry = ToolRegistry()
        tools = registry.list_tools()
        assert len(tools) >= 60


class TestUtils:
    def test_validate_path_safe(self):
        from core.utils import validate_path
        with tempfile.TemporaryDirectory() as tmpdir:
            p = validate_path("test.txt", tmpdir)
            assert str(p).startswith(tmpdir)

    def test_validate_path_traversal(self):
        from core.utils import validate_path
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(PermissionError):
                validate_path("../../../etc/passwd", tmpdir)

    def test_strip_html(self):
        from core.utils import strip_html
        html = "<p>Hello <b>world</b></p><script>alert('xss')</script>"
        result = strip_html(html)
        assert "Hello world" in result
        assert "<script>" not in result
        assert "alert" not in result

    def test_require_optional_import(self):
        from core.utils import require_optional_import
        @require_optional_import("nonexistent_module", "nonexistent")
        def test_func():
            import nonexistent_module
            return "success"
        result = test_func()
        assert "Error" in result
        assert "nonexistent" in result


class TestCalculatorSafety:
    def test_calculator_safe(self):
        from core.tools import ToolRegistry
        registry = ToolRegistry()
        calc = registry.get("calculator")
        result = calc.func("2 + 3")
        assert result == "5"

    def test_calculator_complex(self):
        from core.tools import ToolRegistry
        registry = ToolRegistry()
        calc = registry.get("calculator")
        result = calc.func("2 * (3 + 4)")
        assert result == "14"

    def test_calculator_forbidden(self):
        from core.tools import ToolRegistry
        registry = ToolRegistry()
        calc = registry.get("calculator")
        result = calc.func("__import__('os').system('ls')")
        assert "Error" in result

    def test_calculator_functions(self):
        from core.tools import ToolRegistry
        registry = ToolRegistry()
        calc = registry.get("calculator")
        result = calc.func("sqrt(16)")
        assert result == "4.0"
        result = calc.func("abs(-5)")
        assert result == "5"
