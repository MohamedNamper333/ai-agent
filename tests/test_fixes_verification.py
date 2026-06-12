"""
tests/test_fixes_verification.py

اختبارات التحقق من الإصلاحات العشرة.
شغّل بعد تطبيق كل الإصلاحات:
    python -m pytest tests/test_fixes_verification.py -v
"""
import math
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ══════════════════════════════════════════════════════════════
#  إصلاح 1 — OllamaProvider لا يستخدم model_ref
# ══════════════════════════════════════════════════════════════
class TestFix1_OllamaProviderBug:
    """التحقق أن OllamaProvider لا يرسل model_ref لـ LLM.__init__."""

    def test_init_uses_backend_not_model_ref(self):
        """LLM يجب أن يُستدعى بـ backend='ollama' فقط."""
        with patch("core.llm.ollama_provider.LLM") as MockLLM:
            instance = MagicMock()
            MockLLM.return_value = instance

            from core.llm.ollama_provider import OllamaProvider
            OllamaProvider(model="qwen2.5:7b", url="http://localhost:11434")

            # التحقق: لا يوجد model_ref في args
            call_kwargs = MockLLM.call_args[1] if MockLLM.call_args else {}
            call_args = MockLLM.call_args[0] if MockLLM.call_args else ()
            assert "model_ref" not in call_kwargs, (
                "BUG: LLM يتلقى model_ref — هذا يُسبب TypeError في runtime"
            )

    def test_ollama_attributes_set_correctly(self):
        """التحقق أن _use_ollama و _ollama_model تُعيَّن صحيحاً."""
        with patch("core.llm.ollama_provider.LLM") as MockLLM:
            instance = MagicMock()
            MockLLM.return_value = instance

            from core.llm.ollama_provider import OllamaProvider
            OllamaProvider(model="llama3:8b", url="http://remote:11434")

            # التحقق: الـ attributes تُعيَّن يدوياً
            assert instance._use_ollama is True
            assert instance._ollama_model == "llama3:8b"
            assert instance._ollama_base == "http://remote:11434"

    def test_trailing_slash_stripped_from_url(self):
        """URL يجب أن يُزيل الـ trailing slash."""
        with patch("core.llm.ollama_provider.LLM", MagicMock()):
            from core.llm.ollama_provider import OllamaProvider
            p = OllamaProvider(url="http://localhost:11434/")
            assert not p.url.endswith("/")


# ══════════════════════════════════════════════════════════════
#  إصلاح 2 — requirements.txt شامل
# ══════════════════════════════════════════════════════════════
class TestFix2_Requirements:
    """التحقق أن المكتبات الحرجة مذكورة في requirements.txt."""

    def _get_requirements(self) -> str:
        req = Path("requirements.txt")
        if not req.exists():
            pytest.skip("requirements.txt غير موجود")
        return req.read_text()

    @pytest.mark.parametrize("package", [
        "pandas",
        "numpy",
        "openpyxl",
        "httpx",
        "openai",
        "beautifulsoup4",
        "gunicorn",
        "sentence-transformers",
    ])
    def test_package_in_requirements(self, package):
        content = self._get_requirements()
        assert package.lower() in content.lower(), (
            f"'{package}' مفقود من requirements.txt — يُسبب ImportError في production"
        )


# ══════════════════════════════════════════════════════════════
#  إصلاح 3 — Auth مربوط
# ══════════════════════════════════════════════════════════════
class TestFix3_AuthConnected:
    """التحقق أن web.py يتحقق من Auth على الـ endpoints الحساسة."""

    def test_chat_requires_auth(self):
        from fastapi.testclient import TestClient
        with patch("web.Agent"), patch("web.Retriever"), patch("web.LLM"):
            from web import app
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/chat", json={"message": "hello", "stream": False})
            # يجب أن يُرجع 401 بدون token
            assert resp.status_code in (401, 422), (
                f"GET /chat يجب أن يطلب Auth. رمز الحالة: {resp.status_code}"
            )

    def test_conversations_requires_auth(self):
        from fastapi.testclient import TestClient
        with patch("web.Agent"), patch("web.Retriever"), patch("web.LLM"):
            from web import app
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/conversations")
            assert resp.status_code == 401

    def test_status_is_public(self):
        """GET /status يجب أن يعمل بدون auth."""
        from fastapi.testclient import TestClient
        with patch("web.Agent"), patch("web.Retriever"), patch("web.LLM"):
            from web import app
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/status")
            assert resp.status_code == 200

    def test_auth_init_admin_endpoint_exists(self):
        """POST /auth/init-admin يجب أن يكون موجوداً."""
        from fastapi.testclient import TestClient
        with patch("web.Agent"), patch("web.Retriever"), patch("web.LLM"):
            from web import app
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/auth/init-admin")
            # 200 (أول مرة) أو 400 (admin موجود) — كلاهما صحيح
            assert resp.status_code in (200, 400)


# ══════════════════════════════════════════════════════════════
#  إصلاح 4 — Rate Limiter مفعّل
# ══════════════════════════════════════════════════════════════
class TestFix4_RateLimiter:
    """التحقق أن Rate Limiting يعمل على الـ requests."""

    def test_rate_limit_middleware_returns_429(self):
        """بعد تجاوز الحد، يجب أن يُرجع 429."""
        from core.rate_limiter import RateLimiter
        limiter = RateLimiter(max_requests=2, window_seconds=60)

        assert limiter.is_allowed("test-ip") is True
        assert limiter.is_allowed("test-ip") is True
        assert limiter.is_allowed("test-ip") is False  # حُجب

    def test_rate_limit_header_present(self):
        """X-RateLimit-Remaining يجب أن يُضاف في الـ response."""
        from fastapi.testclient import TestClient
        with patch("web.Agent"), patch("web.Retriever"), patch("web.LLM"):
            from web import app
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/status")
            assert "x-ratelimit-remaining" in resp.headers, (
                "Header X-RateLimit-Remaining مفقود من الـ response"
            )


# ══════════════════════════════════════════════════════════════
#  إصلاح 5 — VectorStore cache
# ══════════════════════════════════════════════════════════════
class TestFix5_VectorStoreCache:
    """التحقق أن numpy cache يعمل ولا يُعاد بناؤه في كل search."""

    def test_cache_built_once_for_repeated_searches(self, tmp_path):
        from rag.vector_store import VectorStore
        store = VectorStore(db_path=str(tmp_path / "v.json"))

        emb = [0.1] * 384
        store.add("doc1", emb, {"source": "test"})
        store.add("doc2", [0.9] * 384, {"source": "test"})

        # البحث الأول — يبني الـ cache
        _ = store.search(emb, top_k=1)
        assert store._cache_valid is True

        # البحث الثاني — يستخدم الـ cache (لا يُعيد البناء)
        matrix_before = id(store._np_matrix)
        _ = store.search(emb, top_k=1)
        assert id(store._np_matrix) == matrix_before, (
            "Cache يُعاد بناؤه في كل بحث — يجب أن يُبقى محفوظاً"
        )

    def test_cache_invalidated_on_add(self, tmp_path):
        from rag.vector_store import VectorStore
        store = VectorStore(db_path=str(tmp_path / "v.json"))

        store.add("doc1", [0.1] * 384)
        _ = store.search([0.1] * 384, top_k=1)
        assert store._cache_valid is True

        # إضافة وثيقة تُبطل الـ cache
        store.add("doc2", [0.9] * 384)
        assert store._cache_valid is False, (
            "Cache يجب أن يُبطَل عند إضافة وثيقة جديدة"
        )

    def test_search_returns_correct_result(self, tmp_path):
        from rag.vector_store import VectorStore
        store = VectorStore(db_path=str(tmp_path / "v.json"))

        store.add("قريب", [1.0, 0.0, 0.0] + [0.0] * 381)
        store.add("بعيد", [0.0, 1.0, 0.0] + [0.0] * 381)

        results = store.search([1.0, 0.0, 0.0] + [0.0] * 381, top_k=1)
        assert len(results) == 1
        assert results[0]["text"] == "قريب"
        assert results[0]["score"] > 0.99


# ══════════════════════════════════════════════════════════════
#  إصلاح 6 — Embedder fallback معنوي
# ══════════════════════════════════════════════════════════════
class TestFix6_EmbedderFallback:
    """التحقق أن fallback embedder ينتج vectors دلالية حقيقية."""

    def _cosine(self, a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        return dot / (na * nb) if na and nb else 0.0

    def test_similar_texts_have_higher_cosine_than_different(self):
        from rag.embedder import Embedder
        e = Embedder()

        v1 = e._tfidf_embed("python programming error bug")
        v2 = e._tfidf_embed("python code error fix")
        v3 = e._tfidf_embed("banana fruit tropical yellow")

        sim_related = self._cosine(v1, v2)
        sim_unrelated = self._cosine(v1, v3)

        assert sim_related > sim_unrelated, (
            f"Fallback غير معنوي: related={sim_related:.3f} <= unrelated={sim_unrelated:.3f}\n"
            "يجب أن تكون النصوص المتشابهة أقرب من غير المتشابهة"
        )

    def test_output_is_normalized(self):
        from rag.embedder import Embedder
        e = Embedder()
        v = e._tfidf_embed("test text for normalization check")
        norm = math.sqrt(sum(x * x for x in v))
        assert abs(norm - 1.0) < 0.01, f"Vector غير مُعيَّر: norm={norm:.4f}"

    def test_output_dimension_is_384(self):
        from rag.embedder import Embedder
        e = Embedder()
        v = e._tfidf_embed("any text here")
        assert len(v) == 384

    def test_empty_text_returns_zeros(self):
        from rag.embedder import Embedder
        e = Embedder()
        v = e._tfidf_embed("")
        assert all(x == 0.0 for x in v)

    def test_same_text_same_embedding(self):
        from rag.embedder import Embedder
        e = Embedder()
        v1 = e._tfidf_embed("python is great")
        v2 = e._tfidf_embed("python is great")
        assert v1 == v2, "نفس النص يجب أن ينتج نفس الـ embedding (deterministic)"


# ══════════════════════════════════════════════════════════════
#  إصلاح 7 — tools_base.py مستقل
# ══════════════════════════════════════════════════════════════
class TestFix7_ToolsBaseSplit:
    """التحقق أن Tool و ToolResult محمولان من tools_base.py."""

    def test_import_from_tools_base(self):
        from core.tools_base import Tool, ToolResult
        assert Tool is not None
        assert ToolResult is not None

    def test_tool_result_works(self):
        from core.tools_base import ToolResult
        r = ToolResult(tool_name="test", success=True, result="output")
        assert bool(r) is True
        assert r.tool_name == "test"

    def test_tool_run_works(self):
        from core.tools_base import Tool
        t = Tool("add", "adds numbers", lambda a, b: a + b)
        result = t.run(a=2, b=3)
        assert result.success is True
        assert result.result == "5"

    def test_tool_error_handling(self):
        from core.tools_base import Tool
        t = Tool("fail", "always fails", lambda: (_ for _ in ()).throw(ValueError("boom")))
        result = t.run()
        assert result.success is False
        assert "boom" in result.error

    def test_backward_compat_import_from_tools(self):
        """from core.tools import Tool, ToolResult يجب أن يعمل بدون تغيير."""
        from core.tools import Tool, ToolResult, ToolRegistry
        assert Tool is not None
        assert ToolResult is not None
        assert ToolRegistry is not None


# ══════════════════════════════════════════════════════════════
#  إصلاح 8 — Docker files موجودة
# ══════════════════════════════════════════════════════════════
class TestFix8_DockerFiles:
    """التحقق من وجود ملفات Docker."""

    @pytest.mark.parametrize("filename", [
        "Dockerfile",
        "docker-compose.yml",
        ".dockerignore",
        "gunicorn.conf.py",
    ])
    def test_file_exists(self, filename):
        assert Path(filename).exists(), (
            f"'{filename}' مفقود — يجب أن يكون موجوداً للـ Docker deployment"
        )

    def test_dockerfile_has_healthcheck(self):
        content = Path("Dockerfile").read_text()
        assert "HEALTHCHECK" in content

    def test_dockerfile_has_nonroot_user(self):
        content = Path("Dockerfile").read_text()
        assert "USER" in content, "Dockerfile يجب أن يُشغَّل بمستخدم غير root"


# ══════════════════════════════════════════════════════════════
#  إصلاح 9 — toggle_rag يستخدم RAG_ENABLED
# ══════════════════════════════════════════════════════════════
class TestFix9_ToggleRag:
    """التحقق أن toggle_rag يستخدم config.RAG_ENABLED لا config.RAG."""

    def test_toggle_rag_uses_rag_enabled(self):
        import config
        from fastapi.testclient import TestClient
        with patch("web.Agent"), patch("web.Retriever"), patch("web.LLM"), \
             patch("web._require_user", return_value=MagicMock()):
            from web import app
            original = getattr(config, "RAG_ENABLED", True)
            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post(
                "/settings/rag",
                headers={"Authorization": "Bearer any-key"}
            )
            # يجب أن يُرجع rag_enabled وليس خطأ
            assert "rag_enabled" in resp.json()
            config.RAG_ENABLED = original  # restore


# ══════════════════════════════════════════════════════════════
#  الاختبار الشامل — كل الـ imports تعمل معاً
# ══════════════════════════════════════════════════════════════
class TestAllImportsTogether:
    """التحقق أن كل التعديلات متوافقة مع بعضها."""

    def test_all_core_imports(self):
        from core.tools_base import Tool, ToolResult
        from core.tools import ToolRegistry
        from core.auth import AuthManager, UserRole
        from core.rate_limiter import TieredRateLimiter
        from core.memory import ConversationMemory
        from core.cache import LRUCache
        from core.telemetry import Telemetry
        assert all([Tool, ToolResult, ToolRegistry, AuthManager, TieredRateLimiter])

    def test_all_rag_imports(self):
        from rag.embedder import Embedder
        from rag.vector_store import VectorStore
        from rag.retriever import Retriever
        assert all([Embedder, VectorStore, Retriever])

    def test_all_llm_imports(self):
        from core.llm import LLMRouter, OllamaProvider, OpenCodeZenProvider
        from core.llm.base import LLMRequest, LLMResponse, ReasoningLevel
        assert all([LLMRouter, OllamaProvider, OpenCodeZenProvider])
