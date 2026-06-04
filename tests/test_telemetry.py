"""Tests for core.telemetry — event tracking, ring buffer, rotation, reporting."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest

from core.telemetry import (
    AgentEvent,
    DEFAULT_LOG_FILE,
    Telemetry,
    is_telemetry_enabled,
)


@pytest.fixture
def tel(tmp_path) -> Telemetry:
    """Telemetry with isolated log_dir under tmp_path."""
    return Telemetry(log_dir=str(tmp_path), max_events=1000, enabled=True)


# ---------- 1 ----------
def test_agent_event_defaults():
    ev = AgentEvent(name="ping")
    assert ev.name == "ping"
    assert ev.status == "ok"
    assert ev.duration_ms == 0
    assert ev.event_id and len(ev.event_id) == 32
    assert ev.ts and "T" in ev.ts
    assert ev.data == {}


# ---------- 2 ----------
def test_agent_event_to_json_is_serializable():
    ev = AgentEvent(name="x", status="error", duration_ms=12.5, data={"k": 1})
    payload = json.loads(ev.to_json())
    assert payload["name"] == "x"
    assert payload["status"] == "error"
    assert payload["duration_ms"] == 12.5
    assert payload["data"] == {"k": 1}
    assert payload["event_id"] == ev.event_id
    assert payload["ts"] == ev.ts
    assert list(payload.keys()) == sorted(payload.keys())


# ---------- 3 ----------
def test_track_context_manager_ok(tel):
    with tel.track("op", user="u1") as ev:
        time.sleep(0.002)
    assert ev.status == "ok"
    assert ev.duration_ms > 0
    assert ev.name == "op"
    assert ev.data == {"user": "u1"}
    assert len(tel.recent(10)) == 1


# ---------- 4 ----------
def test_track_context_manager_error(tel):
    ev = None
    with pytest.raises(RuntimeError):
        with tel.track("op") as ev:
            raise RuntimeError("boom")
    assert ev is not None
    assert ev.status == "error"
    assert "boom" in ev.data.get("error", "")
    recent = tel.recent(10)
    assert recent and recent[-1].status == "error"


# ---------- 5 ----------
def test_event_instant(tel):
    tel.event("n", duration_ms=4.0, data_k="v")
    recent = tel.recent(10)
    assert len(recent) == 1
    assert recent[0].name == "n"
    assert recent[0].duration_ms == 4.0
    assert recent[0].data.get("data_k") == "v"


# ---------- 6 ----------
def test_report_aggregates(tel):
    with tel.track("a"):
        pass
    with tel.track("a"):
        pass
    try:
        with tel.track("b"):
            raise ValueError("x")
    except ValueError:
        pass
    rep = tel.report()
    assert rep["total_events"] == 3
    assert rep["by_name"].get("a") == 2
    assert rep["by_name"].get("b") == 1
    assert rep["errors"] == 1
    assert "min" in rep["duration_ms"]
    assert "p95" in rep["duration_ms"]


# ---------- 7 ----------
def test_recent_limit(tel):
    for i in range(5):
        tel.event(f"e{i}")
    recent = tel.recent(3)
    assert len(recent) == 3
    assert {e.name for e in recent} == {"e2", "e3", "e4"}


# ---------- 8 ----------
def test_ring_buffer_cap(tel):
    tel.max_events = 10
    for i in range(15):
        tel.event(f"e{i}")
    recent = tel.recent(100)
    assert len(recent) == 10
    assert recent[0].name == "e5"
    assert recent[-1].name == "e14"


# ---------- 9 ----------
def test_disabled_via_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_TELEMETRY", "0")
    assert is_telemetry_enabled() is False
    t = Telemetry(log_dir=str(tmp_path), max_events=10, enabled=False)
    with t.track("x"):
        pass
    t.event("y")
    assert t.recent(100) == []
    assert not (tmp_path / DEFAULT_LOG_FILE).exists()


# ---------- 10 ----------
def test_file_rotation(tmp_path, monkeypatch):
    monkeypatch.setattr("core.telemetry.MAX_LOG_BYTES", 150, raising=False)
    t = Telemetry(log_dir=str(tmp_path), max_events=10000, enabled=True)
    for i in range(30):
        t.event(f"event_{i}", data="x" * 5)
    log_path = Path(t.log_dir) / DEFAULT_LOG_FILE
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines
    last_event = json.loads(lines[-1])
    assert last_event["name"] == "event_29"
    assert log_path.stat().st_size <= 400


# ---------- 11 (bonus) ----------
def test_custom_log_dir(tmp_path):
    custom = tmp_path / "nested" / "telemetry"
    t = Telemetry(log_dir=str(custom), max_events=10, enabled=True)
    with t.track("x", k=1):
        pass
    log_path = custom / DEFAULT_LOG_FILE
    assert log_path.exists()
    payload = json.loads(log_path.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert payload["name"] == "x"
    assert payload["data"] == {"k": 1}


# ---------- 12 (bonus) ----------
def test_thread_safety(tel):
    barrier = threading.Barrier(8)

    def worker(i: int) -> None:
        barrier.wait()
        with tel.track("concurrent", i=str(i)):
            pass

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    rep = tel.report()
    assert rep["total_events"] == 8
    assert rep["by_name"].get("concurrent") == 8
    assert rep["errors"] == 0
