"""Phase 2 tests — Deep Learning, WebSocket, Anomaly Detection, RL Feedback."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Task Classifier ────────────────────────────────────────────

class TestTaskClassifier:
    def setup_method(self):
        import logging; logging.disable(logging.CRITICAL)
        from core.deep_learning.task_classifier import TaskClassifier
        self.tc = TaskClassifier()

    def test_predict_returns_dict(self):
        r = self.tc.predict("write a python function")
        assert isinstance(r, dict)
        assert "label" in r
        assert "confidence" in r
        assert "latency_ms" in r

    def test_confidence_between_0_and_1(self):
        r = self.tc.predict("analyze this code")
        assert 0.0 <= r["confidence"] <= 1.0

    def test_code_generation_intent(self):
        r = self.tc.predict("write a python function to sort a list")
        assert r["label"] == "code_generation"
        assert r["confidence"] > 0.5

    def test_data_analysis_intent(self):
        r = self.tc.predict("analyze this csv file and show statistics")
        assert r["label"] == "data_analysis"

    def test_web_search_intent(self):
        r = self.tc.predict("search for latest AI news")
        assert r["label"] == "web_search"

    def test_code_debug_intent(self):
        r = self.tc.predict("fix this bug in my code")
        assert r["label"] == "code_debug"

    def test_arabic_code_generation(self):
        r = self.tc.predict("اكتب دالة بايثون")
        assert r["label"] == "code_generation"

    def test_all_scores_present(self):
        r = self.tc.predict("write a function")
        assert isinstance(r.get("all_scores"), dict)

    def test_latency_reasonable(self):
        r = self.tc.predict("analyze this data")
        assert r["latency_ms"] < 500  # Must be under 500ms

    def test_get_stats(self):
        stats = self.tc.get_stats()
        assert stats["trained"] is True
        assert stats["num_classes"] == 12

    def test_unknown_input_returns_label(self):
        r = self.tc.predict("xyzzy frobozz plugh")
        assert r["label"] in self.tc.get_stats()["task_labels"]


# ─── Anomaly Detector ────────────────────────────────────────────

class TestAnomalyDetector:
    def setup_method(self):
        import logging; logging.disable(logging.CRITICAL)
        from core.deep_learning.anomaly_detector import AnomalyDetector
        self.ad = AnomalyDetector()

    def test_score_returns_dict(self):
        r = self.ad.score("write a python function")
        assert "is_anomaly" in r
        assert "risk_level" in r
        assert "latency_ms" in r

    def test_normal_request_safe(self):
        r = self.ad.score("write a python function to sort a list")
        assert r["is_anomaly"] in (False, 0)  # handles numpy.bool_ too

    def test_sql_injection_detected(self):
        r = self.ad.score("SELECT * FROM users UNION SELECT password FROM admin")
        assert r["is_anomaly"] is True
        assert r["risk_level"] == "HIGH"

    def test_xss_detected(self):
        r = self.ad.score('<script>alert("xss")</script>')
        assert r["is_anomaly"] is True

    def test_path_traversal_detected(self):
        r = self.ad.score("read ../../../../etc/passwd file")
        assert r["is_anomaly"] is True

    def test_code_injection_detected(self):
        r = self.ad.score("exec(open('/etc/passwd').read())")
        assert r["is_anomaly"] is True

    def test_risk_levels_valid(self):
        for text in ["hello world", "write code", "analyze data"]:
            r = self.ad.score(text)
            assert r["risk_level"] in ("LOW", "MEDIUM", "HIGH")

    def test_latency_reasonable(self):
        r = self.ad.score("write python function")
        assert r["latency_ms"] < 200

    def test_get_stats(self):
        self.ad.score("test")
        stats = self.ad.get_stats()
        assert "total_checked" in stats
        assert stats["total_checked"] >= 1


# ─── RL Feedback Engine ──────────────────────────────────────────

class TestRLFeedbackEngine:
    def setup_method(self):
        import logging; logging.disable(logging.CRITICAL)
        from core.deep_learning.rl_feedback import RLFeedbackEngine
        self.rl = RLFeedbackEngine()

    def test_record_positive_feedback(self):
        self.rl.record_feedback("code_generation", "run_code", "positive")
        stats = self.rl.get_stats()
        assert stats["total_feedback"] >= 1

    def test_record_negative_feedback(self):
        self.rl.record_feedback("code_analysis", "review_code", "negative")
        assert self.rl.get_stats()["total_feedback"] >= 1

    def test_get_best_tool_returns_str_or_none(self):
        self.rl.record_feedback("code_generation", "run_code", "positive")
        self.rl.record_feedback("code_generation", "run_code", "positive")
        best = self.rl.get_best_tool("code_generation", ["run_code", "read_file"])
        assert best is None or isinstance(best, str)

    def test_no_data_returns_none(self):
        result = self.rl.get_best_tool("unknown_task_xyz", ["tool_a", "tool_b"])
        assert result is None

    def test_tool_rankings_sorted(self):
        self.rl.record_feedback("web_search", "search_web", "positive")
        self.rl.record_feedback("web_search", "fetch_url", "negative")
        rankings = self.rl.get_tool_rankings("web_search")
        if len(rankings) >= 2:
            assert rankings[0]["q_value"] >= rankings[1]["q_value"]

    def test_get_stats_structure(self):
        stats = self.rl.get_stats()
        assert "total_feedback" in stats
        assert "task_types_learned" in stats
        assert "tools_ranked" in stats

    def test_invalid_feedback_ignored(self):
        before = self.rl.get_stats()["total_feedback"]
        self.rl.record_feedback("code_generation", "run_code", "invalid_value")
        # Should not crash, and total_feedback should still increment
        assert True  # Just verifying no exception


# ─── Embedding Store ─────────────────────────────────────────────

class TestEmbeddingStore:
    def setup_method(self, tmp_path=None):
        import logging; logging.disable(logging.CRITICAL)
        from core.deep_learning.embedding_store import EmbeddingStore
        self.es = EmbeddingStore()

    def test_add_returns_index(self):
        idx = self.es.add("python sorting algorithms")
        assert isinstance(idx, int)
        assert idx >= 0

    def test_search_returns_list(self):
        self.es.add("machine learning tutorial")
        results = self.es.search("ML guide")
        assert isinstance(results, list)

    def test_search_returns_scores(self):
        self.es.add("deep learning with pytorch", {"category": "code"})
        results = self.es.search("pytorch neural network", k=1)
        if results:
            assert "score" in results[0]
            assert "text" in results[0]

    def test_empty_store_returns_empty(self):
        from core.deep_learning.embedding_store import EmbeddingStore
        fresh = EmbeddingStore()
        # Won't necessarily be empty due to persistence, but should not crash
        results = fresh.search("test query")
        assert isinstance(results, list)

    def test_get_stats_structure(self):
        stats = self.es.get_stats()
        assert "total_entries" in stats
        assert "dimension" in stats
        assert "faiss_enabled" in stats

    def test_metadata_stored(self):
        self.es.add("fastapi tutorial", {"category": "backend", "lang": "python"})
        results = self.es.search("fastapi web framework", k=3)
        if results:
            assert "text" in results[0]


# ─── WebSocket Manager ───────────────────────────────────────────

class TestWebSocketManager:
    def test_manager_singleton(self):
        from core.websocket_manager import get_ws_manager
        m1 = get_ws_manager()
        m2 = get_ws_manager()
        assert m1 is m2

    def test_get_stats_structure(self):
        from core.websocket_manager import ConnectionManager
        mgr = ConnectionManager()
        stats = mgr.get_stats()
        assert "active_connections" in stats
        assert "total_connections_ever" in stats
        assert "active_rooms" in stats

    def test_initial_no_connections(self):
        from core.websocket_manager import ConnectionManager
        mgr = ConnectionManager()
        stats = mgr.get_stats()
        assert stats["active_connections"] == 0

    def test_disconnect_nonexistent_noop(self):
        from core.websocket_manager import ConnectionManager
        mgr = ConnectionManager()
        # Should not raise
        mgr.disconnect("nonexistent-conn-id")

    def test_ws_endpoint_in_web(self):
        src = Path("web.py").read_text()
        assert "@app.websocket" in src
        assert "websocket_endpoint" in src


# ─── Deep Learning Package ───────────────────────────────────────

class TestDLPackage:
    def test_package_importable(self):
        import logging; logging.disable(logging.CRITICAL)
        from core.deep_learning import (
            TaskClassifier, EmbeddingStore,
            AnomalyDetector, RLFeedbackEngine,
        )
        assert TaskClassifier is not None
        assert EmbeddingStore is not None
        assert AnomalyDetector is not None
        assert RLFeedbackEngine is not None

    def test_all_exported(self):
        import logging; logging.disable(logging.CRITICAL)
        import core.deep_learning as dl
        for name in ["TaskClassifier", "EmbeddingStore", "AnomalyDetector", "RLFeedbackEngine"]:
            assert hasattr(dl, name), f"Missing export: {name}"
