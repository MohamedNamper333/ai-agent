"""Tests for tools/scheduler.py"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from tools.scheduler import TaskScheduler


@pytest.fixture
def scheduler(tmp_path):
    """Create a fresh TaskScheduler with a temporary DB path."""
    db = tmp_path / "tasks.json"
    return TaskScheduler(db_path=str(db))


@pytest.fixture
def scheduler_with_tasks(scheduler):
    """Scheduler pre-loaded with one task of each type."""
    scheduler.add_task("every5", "do something", "5", task_type="interval")
    scheduler.add_task("morning", "wake up", "08:30", task_type="daily")
    scheduler.add_task("oncejob", "one off", "30", task_type="once")
    return scheduler


# ── Initialization ───────────────────────────────────────────────────────────

class TestInit:
    def test_default_db_path(self, tmp_path):
        s = TaskScheduler()
        assert s.tasks == []
        assert s._running is False
        assert s._callback is None

    def test_custom_db_path(self, tmp_path):
        db = tmp_path / "custom.json"
        s = TaskScheduler(db_path=str(db))
        assert s.db_path == str(db)

    def test_initial_state(self, scheduler):
        assert scheduler.tasks == []
        assert scheduler._history == []
        assert scheduler._loaded is False


# ── Load ─────────────────────────────────────────────────────────────────────

class TestLoad:
    def test_load_from_nonexistent_file(self, tmp_path):
        s = TaskScheduler(db_path=str(tmp_path / "nope.json"))
        s.load()
        assert s.tasks == []
        assert s._loaded is True

    def test_load_creates_tasks(self, tmp_path):
        db = tmp_path / "tasks.json"
        data = {
            "tasks": [{"id": "t1", "name": "saved"}],
            "history": [{"task_id": "t1"}],
        }
        db.write_text(json.dumps(data), encoding="utf-8")

        s = TaskScheduler(db_path=str(db))
        s.load()
        assert len(s.tasks) == 1
        assert s.tasks[0]["name"] == "saved"
        assert len(s._history) == 1

    def test_load_old_format_list(self, tmp_path):
        db = tmp_path / "tasks.json"
        db.write_text(json.dumps([{"id": "t1"}]), encoding="utf-8")

        s = TaskScheduler(db_path=str(db))
        s.load()
        assert len(s.tasks) == 1

    def test_load_corrupt_file(self, tmp_path):
        db = tmp_path / "tasks.json"
        db.write_text("NOT JSON", encoding="utf-8")

        s = TaskScheduler(db_path=str(db))
        s.load()
        assert s.tasks == []

    def test_load_called_twice_is_noop(self, scheduler):
        scheduler.load()
        scheduler.add_task("x", "p", "10")
        # second load should not reload from disk (already _loaded)
        scheduler.load()
        assert len(scheduler.tasks) == 1


# ── Add Task ─────────────────────────────────────────────────────────────────

class TestAddTask:
    def test_add_interval_task(self, scheduler):
        task = scheduler.add_task("poll", "check emails", "15", task_type="interval")
        assert task["name"] == "poll"
        assert task["task_type"] == "interval"
        assert task["schedule"] == "15"
        assert task["enabled"] is True
        assert task["run_count"] == 0
        assert task["id"].startswith("task_")

    def test_add_daily_task(self, scheduler):
        task = scheduler.add_task("morning", "summary", "07:00", task_type="daily")
        assert task["task_type"] == "daily"
        # next_run should be a valid ISO timestamp
        dt = datetime.fromisoformat(task["next_run"])
        assert dt > datetime.now()

    def test_add_once_task(self, scheduler):
        task = scheduler.add_task("backup", "run backup", "60", task_type="once")
        assert task["task_type"] == "once"
        assert task["next_run"] != ""

    def test_invalid_task_type(self, scheduler):
        result = scheduler.add_task("bad", "p", "10", task_type="invalid")
        assert "error" in result

    def test_task_persisted_to_disk(self, scheduler):
        scheduler.add_task("persist", "p", "5")
        # reload from disk
        s2 = TaskScheduler(db_path=scheduler.db_path)
        s2.load()
        assert len(s2.tasks) == 1
        assert s2.tasks[0]["name"] == "persist"

    def test_multiple_tasks(self, scheduler):
        scheduler.add_task("a", "pa", "5")
        scheduler.add_task("b", "pb", "10")
        scheduler.add_task("c", "pc", "15")
        assert len(scheduler.tasks) == 3

    def test_task_ids_unique(self, scheduler):
        t1 = scheduler.add_task("a", "p", "5")
        time.sleep(0.01)
        t2 = scheduler.add_task("b", "p", "5")
        assert t1["id"] != t2["id"]


# ── Remove Task ──────────────────────────────────────────────────────────────

class TestRemoveTask:
    def test_remove_existing(self, scheduler_with_tasks):
        tid = scheduler_with_tasks.tasks[0]["id"]
        assert scheduler_with_tasks.remove_task(tid) is True
        assert len(scheduler_with_tasks.tasks) == 2

    def test_remove_nonexistent(self, scheduler):
        assert scheduler.remove_task("task_999") is False

    def test_remove_persists(self, scheduler):
        t = scheduler.add_task("x", "p", "5")
        scheduler.remove_task(t["id"])
        s2 = TaskScheduler(db_path=scheduler.db_path)
        s2.load()
        assert len(s2.tasks) == 0


# ── Enable / Disable ─────────────────────────────────────────────────────────

class TestEnableDisable:
    def test_disable_task(self, scheduler_with_tasks):
        tid = scheduler_with_tasks.tasks[0]["id"]
        assert scheduler_with_tasks.disable_task(tid) is True
        assert scheduler_with_tasks.tasks[0]["enabled"] is False

    def test_enable_task(self, scheduler_with_tasks):
        tid = scheduler_with_tasks.tasks[0]["id"]
        scheduler_with_tasks.disable_task(tid)
        assert scheduler_with_tasks.enable_task(tid) is True
        assert scheduler_with_tasks.tasks[0]["enabled"] is True

    def test_disable_nonexistent(self, scheduler):
        assert scheduler.disable_task("bad_id") is False

    def test_enable_nonexistent(self, scheduler):
        assert scheduler.enable_task("bad_id") is False


# ── List Tasks ───────────────────────────────────────────────────────────────

class TestListTasks:
    def test_list_empty(self, scheduler):
        assert scheduler.list_tasks() == []

    def test_list_returns_copy(self, scheduler_with_tasks):
        tasks = scheduler_with_tasks.list_tasks()
        tasks.clear()
        assert len(scheduler_with_tasks.tasks) == 3

    def test_list_with_tasks(self, scheduler_with_tasks):
        result = scheduler_with_tasks.list_tasks()
        assert len(result) == 3
        names = {t["name"] for t in result}
        assert names == {"every5", "morning", "oncejob"}


# ── Get Task ─────────────────────────────────────────────────────────────────

class TestGetTask:
    def test_get_existing(self, scheduler_with_tasks):
        tid = scheduler_with_tasks.tasks[0]["id"]
        task = scheduler_with_tasks.get_task(tid)
        assert task is not None
        assert task["id"] == tid

    def test_get_returns_copy(self, scheduler_with_tasks):
        tid = scheduler_with_tasks.tasks[0]["id"]
        task = scheduler_with_tasks.get_task(tid)
        task["name"] = "modified"
        assert scheduler_with_tasks.get_task(tid)["name"] != "modified"

    def test_get_nonexistent(self, scheduler):
        assert scheduler.get_task("nope") is None


# ── History ──────────────────────────────────────────────────────────────────

class TestHistory:
    def test_empty_history(self, scheduler):
        assert scheduler.get_history() == []

    def test_history_limit(self, scheduler):
        scheduler._history = [{"i": i} for i in range(50)]
        result = scheduler.get_history(limit=5)
        assert len(result) == 5
        assert result[0]["i"] == 45

    def test_history_default_limit(self, scheduler):
        scheduler._history = [{"i": i} for i in range(200)]
        result = scheduler.get_history()
        assert len(result) == 20


# ── Callback / Execution ─────────────────────────────────────────────────────

class TestCallback:
    def test_set_callback(self, scheduler):
        cb = MagicMock(return_value="ok")
        scheduler.set_callback(cb)
        assert scheduler._callback is cb

    def test_callback_called_on_due_task(self, scheduler):
        cb = MagicMock(return_value="done")
        scheduler.set_callback(cb)
        scheduler._running = False  # ensure no background loop

        task = scheduler.add_task("instant", "test prompt", "0", task_type="interval")
        # Force next_run to the past so task is due
        task["next_run"] = (datetime.now() - timedelta(seconds=1)).isoformat()
        scheduler._save()

        # Manually trigger one run cycle iteration (without the sleep loop)
        scheduler.load()
        now = datetime.now()
        for t in scheduler.tasks:
            if not t.get("enabled", True):
                continue
            next_run = t.get("next_run", "")
            if next_run and now >= datetime.fromisoformat(next_run):
                result = cb(t["name"], t["prompt"])
                t["last_run"] = now.isoformat()
                t["run_count"] = t.get("run_count", 0) + 1
                scheduler._history.append({
                    "task_id": t["id"],
                    "task_name": t["name"],
                    "result": str(result),
                    "timestamp": now.isoformat(),
                    "status": "success",
                })
                if t["task_type"] == "once":
                    t["enabled"] = False
                scheduler._save()

        cb.assert_called_once_with("instant", "test prompt")
        assert scheduler.tasks[0]["run_count"] == 1
        assert len(scheduler._history) == 1

    def test_once_task_disabled_after_run(self, scheduler):
        cb = MagicMock(return_value="ok")
        scheduler.set_callback(cb)

        task = scheduler.add_task("one", "do it", "30", task_type="once")
        task["next_run"] = (datetime.now() - timedelta(seconds=1)).isoformat()
        scheduler._save()

        # Simulate run
        now = datetime.now()
        for t in scheduler.tasks:
            nr = t.get("next_run", "")
            if nr and now >= datetime.fromisoformat(nr):
                cb(t["name"], t["prompt"])
                t["last_run"] = now.isoformat()
                t["run_count"] = 1
                if t["task_type"] == "once":
                    t["enabled"] = False
                    t["next_run"] = ""
                scheduler._save()

        assert scheduler.tasks[0]["enabled"] is False
        assert scheduler.tasks[0]["next_run"] == ""

    def test_callback_exception_recorded(self, scheduler):
        def bad_cb(name, prompt):
            raise ValueError("boom")

        scheduler.set_callback(bad_cb)

        task = scheduler.add_task("fail", "p", "30", task_type="once")
        task["next_run"] = (datetime.now() - timedelta(seconds=1)).isoformat()
        scheduler._save()

        now = datetime.now()
        for t in scheduler.tasks:
            nr = t.get("next_run", "")
            if nr and now >= datetime.fromisoformat(nr):
                try:
                    bad_cb(t["name"], t["prompt"])
                    status = "success"
                    result = ""
                except Exception as e:
                    scheduler._history.append({
                        "task_id": t["id"],
                        "task_name": t["name"],
                        "error": str(e),
                        "timestamp": now.isoformat(),
                        "status": "failed",
                    })
                    status = "failed"
                t["last_run"] = now.isoformat()
                t["run_count"] = 1
                if t["task_type"] == "once":
                    t["enabled"] = False
                scheduler._save()

        assert len(scheduler._history) == 1
        assert scheduler._history[0]["status"] == "failed"
        assert "boom" in scheduler._history[0]["error"]


# ── Start / Stop ─────────────────────────────────────────────────────────────

class TestStartStop:
    def test_start_creates_thread(self, scheduler):
        scheduler.start()
        assert scheduler._running is True
        assert scheduler._thread is not None
        assert scheduler._thread.is_alive()
        scheduler.stop()

    def test_start_twice_is_noop(self, scheduler):
        scheduler.start()
        t1 = scheduler._thread
        scheduler.start()  # should not create a second thread
        assert scheduler._thread is t1
        scheduler.stop()

    def test_stop(self, scheduler):
        scheduler.start()
        scheduler.stop()
        time.sleep(0.1)
        assert scheduler._running is False


# ── Calculate Next Run ───────────────────────────────────────────────────────

class TestCalculateNextRun:
    def test_interval_valid(self, scheduler):
        result = scheduler._calculate_next_run("30", "interval")
        dt = datetime.fromisoformat(result)
        diff = (dt - datetime.now()).total_seconds()
        assert 29 * 60 <= diff <= 31 * 60

    def test_interval_invalid(self, scheduler):
        result = scheduler._calculate_next_run("abc", "interval")
        dt = datetime.fromisoformat(result)
        diff = (dt - datetime.now()).total_seconds()
        assert 55 * 60 <= diff <= 65 * 60

    def test_daily_valid(self, scheduler):
        result = scheduler._calculate_next_run("14:30", "daily")
        dt = datetime.fromisoformat(result)
        assert dt.hour == 14 and dt.minute == 30

    def test_daily_past_time_tomorrow(self, scheduler):
        # If current time is past 08:00, next run should be tomorrow
        now = datetime.now()
        if now.hour >= 8:
            result = scheduler._calculate_next_run("08:00", "daily")
            dt = datetime.fromisoformat(result)
            assert dt.date() == (now + timedelta(days=1)).date()

    def test_once(self, scheduler):
        result = scheduler._calculate_next_run("anything", "once")
        dt = datetime.fromisoformat(result)
        diff = (dt - datetime.now()).total_seconds()
        assert 0 <= diff <= 10

    def test_unknown_type_fallback(self, scheduler):
        result = scheduler._calculate_next_run("10", "weird")
        dt = datetime.fromisoformat(result)
        diff = (dt - datetime.now()).total_seconds()
        assert 55 * 60 <= diff <= 65 * 60


# ── Persistence / Save ──────────────────────────────────────────────────────

class TestPersistence:
    def test_save_writes_json(self, scheduler):
        scheduler.add_task("saved", "p", "5")
        assert Path(scheduler.db_path).exists()
        data = json.loads(Path(scheduler.db_path).read_text(encoding="utf-8"))
        assert "tasks" in data
        assert "history" in data
        assert len(data["tasks"]) == 1

    def test_history_persisted(self, scheduler):
        scheduler._history.append({"task_id": "x", "status": "ok"})
        scheduler._save()
        s2 = TaskScheduler(db_path=scheduler.db_path)
        s2.load()
        assert len(s2._history) == 1

    def test_full_roundtrip(self, tmp_path):
        db = tmp_path / "rt.json"
        s1 = TaskScheduler(db_path=str(db))
        s1.add_task("rt1", "p1", "10", task_type="interval")
        s1.add_task("rt2", "p2", "09:00", task_type="daily")
        s1.disable_task(s1.tasks[0]["id"])
        s1._history.append({"task_id": "rt1", "status": "success"})
        s1._save()

        s2 = TaskScheduler(db_path=str(db))
        s2.load()
        assert len(s2.tasks) == 2
        assert s2.tasks[0]["enabled"] is False
        assert s2.tasks[1]["enabled"] is True
        assert len(s2._history) == 1
