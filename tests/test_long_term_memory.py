import json
import tempfile
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.long_term_memory import LongTermMemory


def _make_ltm(tmp_path):
    return LongTermMemory(db_path=str(tmp_path / "test_memory.json"))


class TestInitialization:
    def test_default_path(self):
        ltm = LongTermMemory()
        assert ltm.summaries == []
        assert ltm._loaded is False

    def test_custom_path(self, tmp_path):
        ltm = _make_ltm(tmp_path)
        assert ltm.db_path == str(tmp_path / "test_memory.json")
        assert ltm.summaries == []

    def test_load_creates_empty_on_missing_file(self, tmp_path):
        ltm = _make_ltm(tmp_path)
        ltm.load()
        assert ltm.summaries == []
        assert ltm._loaded is True


class TestAddSummary:
    def test_add_single(self, tmp_path):
        ltm = _make_ltm(tmp_path)
        ltm.add_summary("conv-1", "Discussed Python decorators", topics=["python", "decorators"])
        assert len(ltm.summaries) == 1
        assert ltm.summaries[0]["conversation_id"] == "conv-1"
        assert ltm.summaries[0]["summary"] == "Discussed Python decorators"
        assert ltm.summaries[0]["topics"] == ["python", "decorators"]
        assert "timestamp" in ltm.summaries[0]

    def test_add_persists_to_disk(self, tmp_path):
        ltm = _make_ltm(tmp_path)
        ltm.add_summary("conv-1", "Hello world", topics=["greet"])
        with open(ltm.db_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["conversation_id"] == "conv-1"

    def test_add_multiple_independent(self, tmp_path):
        ltm1 = _make_ltm(tmp_path)
        ltm1.add_summary("c1", "First", topics=["a"])
        ltm2 = _make_ltm(tmp_path)
        ltm2.add_summary("c2", "Second", topics=["b"])
        assert len(ltm2.summaries) == 2


class TestSearch:
    def _populate(self, tmp_path):
        ltm = _make_ltm(tmp_path)
        ltm.add_summary("c1", "Python list comprehension tutorial", topics=["python", "lists"])
        ltm.add_summary("c2", "JavaScript async await patterns", topics=["javascript", "async"])
        ltm.add_summary("c3", "Python decorator advanced guide", topics=["python", "decorators"])
        return ltm

    def test_search_matching_keywords(self, tmp_path):
        ltm = self._populate(tmp_path)
        results = ltm.search("python")
        assert len(results) >= 1
        summaries = [s["summary"] for _, s in results]
        assert any("Python" in s for s in summaries)

    def test_search_no_keyword_overlap_scores_lower(self, tmp_path):
        ltm = self._populate(tmp_path)
        related = ltm.search("python")
        unrelated = ltm.search("kubernetes docker")
        related_score = related[0][0]
        unrelated_score = max(s for s, _ in unrelated)
        assert unrelated_score < related_score

    def test_search_respects_top_k(self, tmp_path):
        ltm = self._populate(tmp_path)
        results = ltm.search("python", top_k=1)
        assert len(results) <= 1

    def test_search_empty_db(self, tmp_path):
        ltm = _make_ltm(tmp_path)
        results = ltm.search("anything")
        assert results == []


class TestMultipleSummaries:
    def test_search_scores_topic_match_higher(self, tmp_path):
        ltm = _make_ltm(tmp_path)
        ltm.add_summary("c1", "A brief mention of python", topics=["java"])
        ltm.add_summary("c2", "Deep dive into python testing", topics=["python"])
        results = ltm.search("python")
        assert len(results) >= 2
        assert results[0][1]["conversation_id"] == "c2"


class TestRecallContext:
    def test_recall_with_results(self, tmp_path):
        ltm = _make_ltm(tmp_path)
        ltm.add_summary("c1", "Reviewed project architecture", topics=["architecture"])
        ctx = ltm.get_context("architecture")
        assert "[Long-term memory recall]" in ctx
        assert "project architecture" in ctx

    def test_recall_no_results(self, tmp_path):
        ltm = _make_ltm(tmp_path)
        ctx = ltm.get_context("xyz_nonexistent")
        assert ctx == ""


class TestDeleteSummary:
    def test_delete_existing(self, tmp_path):
        ltm = _make_ltm(tmp_path)
        ltm.add_summary("c1", "To delete", topics=["x"])
        assert ltm.delete_summary("c1") is True
        assert len(ltm.summaries) == 0

    def test_delete_nonexistent(self, tmp_path):
        ltm = _make_ltm(tmp_path)
        ltm.add_summary("c1", "Keep me", topics=["x"])
        assert ltm.delete_summary("nope") is False
        assert len(ltm.summaries) == 1


class TestGetStats:
    def test_empty_stats(self, tmp_path):
        ltm = _make_ltm(tmp_path)
        stats = ltm.get_stats()
        assert stats["total_summaries"] == 0
        assert stats["unique_topics"] == 0
        assert stats["oldest"] == ""
        assert stats["newest"] == ""

    def test_populated_stats(self, tmp_path):
        ltm = _make_ltm(tmp_path)
        ltm.add_summary("c1", "First", topics=["a", "b"])
        ltm.add_summary("c2", "Second", topics=["b", "c"])
        stats = ltm.get_stats()
        assert stats["total_summaries"] == 2
        assert stats["unique_topics"] == 3
        assert stats["oldest"] != ""
        assert stats["newest"] != ""


class TestReloadBehavior:
    def test_loaded_flag_prevents_reload(self, tmp_path):
        ltm = _make_ltm(tmp_path)
        ltm.add_summary("c1", "Initial", topics=["a"])
        ltm._loaded = False
        ltm.load()
        assert len(ltm.summaries) == 1
