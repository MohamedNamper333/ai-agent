"""Tests for core.telemetry.

Covers:
  - AgentEvent serialization (.to_dict, asdict)
  - Telemetry lifecycle (init defaults, session_id, output_path, buffer_size, enabled)
  - .record() sync API (ok, error, disabled, data defaults)
  - .track() context manager (yields event, measures duration, captures exceptions,
    logs finalization failures but does not break)
  - JSONL persistence (file written, one event per line, valid JSON, encoding)
  - .flush() / buffer_size auto-flush
  - .get_session_events() (lock-safe list copy)
  - .report() (empty + populated, by_type, by_status, avg_duration_ms, errors[:10])
  - .reset() and .close()
  - Error resilience (disabled, _ensure_path failure, flush failure,
    record failure, track finalization failure)
  - _new_session_id() format
  - Thread safety smoke test (concurrent record)
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.telemetry import (
    AgentEvent,
    Telemetry,
    TELEMETRY_DIR,
    TELEMETRY_FILE,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_telemetry_path(tmp_path: Path) -> Path:
    """Provide a fresh JSONL path under tmp_path for each test."""
    return tmp_path / "telemetry.jsonl"


@pytest.fixture
def telemetry(tmp_telemetry_path: Path) -> Telemetry:
    """Build a Telemetry instance writing to a tmp file (enabled, buffer_size=2)."""
    return Telemetry(
        session_id="test-session",
        output_path=tmp_telemetry_path,
        buffer_size=2,
        enabled=True,
    )


# ---------------------------------------------------------------------------
# AgentEvent
# ---------------------------------------------------------------------------


class TestAgentEvent:
    """AgentEvent dataclass + .to_dict() serialization."""

    def test_default_field_values(self) -> None:
        event = AgentEvent(
            event_id="abc",
            event_type="x",
            timestamp=1.0,
            session_id="s",
        )
        assert event.duration_ms == 0.0
        assert event.data == {}
        assert event.status == "ok"
        assert event.error is None

    def test_data_field_uses_default_factory(self) -> None:
        e1 = AgentEvent(event_id="1", event_type="t", timestamp=0.0, session_id="s")
        e2 = AgentEvent(event_id="2", event_type="t", timestamp=0.0, session_id="s")
        e1.data["x"] = 1
        assert e2.data == {}  # independent mutable default

    def test_to_dict_returns_asdict(self) -> None:
        event = AgentEvent(
            event_id="id-1",
            event_type="llm.generate",
            timestamp=123.456,
            session_id="s-1",
            duration_ms=12.5,
            data={"model": "qwen2.5:7b"},
            status="error",
            error="TimeoutError: too slow",
        )
        d = event.to_dict()
        assert d == {
            "event_id": "id-1",
            "event_type": "llm.generate",
            "timestamp": 123.456,
            "session_id": "s-1",
            "duration_ms": 12.5,
            "data": {"model": "qwen2.5:7b"},
            "status": "error",
            "error": "TimeoutError: too slow",
        }

    def test_to_dict_keys_match_fields(self) -> None:
        event = AgentEvent(
            event_id="x", event_type="t", timestamp=0.0, session_id="s"
        )
        assert set(event.to_dict().keys()) == {
            "event_id",
            "event_type",
            "timestamp",
            "session_id",
            "duration_ms",
            "data",
            "status",
            "error",
        }


# ---------------------------------------------------------------------------
# Telemetry __init__
# ---------------------------------------------------------------------------


class TestTelemetryInit:
    """Telemetry instantiation and default behaviour."""

    def test_session_id_explicit(self, tmp_telemetry_path: Path) -> None:
        t = Telemetry(session_id="abc", output_path=tmp_telemetry_path)
        assert t.session_id == "abc"

    def test_session_id_auto_generated(self, tmp_telemetry_path: Path) -> None:
        t = Telemetry(output_path=tmp_telemetry_path)
        assert t.session_id.startswith("session-")
        assert len(t.session_id) > len("session-")

    def test_session_ids_unique(self, tmp_telemetry_path: Path) -> None:
        ids = {
            Telemetry(output_path=tmp_telemetry_path).session_id for _ in range(5)
        }
        assert len(ids) == 5

    def test_output_path_explicit(self, tmp_telemetry_path: Path) -> None:
        t = Telemetry(output_path=tmp_telemetry_path)
        assert t.output_path == tmp_telemetry_path

    def test_output_path_default_is_module_constant(
        self, tmp_telemetry_path: Path
    ) -> None:
        with patch("core.telemetry.TELEMETRY_FILE", tmp_telemetry_path):
            t = Telemetry()
        assert t.output_path == tmp_telemetry_path

    def test_buffer_size_floored_to_one(self, tmp_telemetry_path: Path) -> None:
        t = Telemetry(output_path=tmp_telemetry_path, buffer_size=0)
        assert t.buffer_size == 1
        t2 = Telemetry(output_path=tmp_telemetry_path, buffer_size=-5)
        assert t2.buffer_size == 1

    def test_buffer_size_coerced_to_int(self, tmp_telemetry_path: Path) -> None:
        t = Telemetry(output_path=tmp_telemetry_path, buffer_size=3.0)
        assert t.buffer_size == 3
        assert isinstance(t.buffer_size, int)

    def test_enabled_default_true(self, tmp_telemetry_path: Path) -> None:
        t = Telemetry(output_path=tmp_telemetry_path)
        assert t.enabled is True

    def test_disabled_skips_path_creation(
        self, tmp_telemetry_path: Path
    ) -> None:
        tmp_path_unused = tmp_telemetry_path.parent / "nope.jsonl"
        t = Telemetry(output_path=tmp_path_unused, enabled=False)
        assert t.enabled is False
        # file was not created because _ensure_path not called
        assert not tmp_path_unused.exists()

    def test_enabled_creates_parent_directory(
        self, tmp_telemetry_path: Path
    ) -> None:
        nested = tmp_telemetry_path.parent / "deep" / "nest" / "t.jsonl"
        Telemetry(output_path=nested)
        assert nested.parent.exists()


# ---------------------------------------------------------------------------
# record()
# ---------------------------------------------------------------------------


class TestRecord:
    """Telemetry.record() synchronous event capture."""

    def test_record_returns_agent_event(self, telemetry: Telemetry) -> None:
        ev = telemetry.record("llm.generate", data={"model": "qwen2.5:7b"})
        assert isinstance(ev, AgentEvent)
        assert ev.event_type == "llm.generate"
        assert ev.data == {"model": "qwen2.5:7b"}
        assert ev.status == "ok"
        assert ev.error is None

    def test_record_default_data_is_empty_dict(
        self, telemetry: Telemetry
    ) -> None:
        ev = telemetry.record("simple")
        assert ev is not None
        assert ev.data == {}

    def test_record_none_data_becomes_empty_dict(
        self, telemetry: Telemetry
    ) -> None:
        ev = telemetry.record("simple", data=None)
        assert ev is not None
        assert ev.data == {}

    def test_record_event_id_is_uuid_hex(self, telemetry: Telemetry) -> None:
        ev = telemetry.record("x")
        assert ev is not None
        assert len(ev.event_id) == 32
        int(ev.event_id, 16)  # parseable as hex

    def test_record_unique_event_ids(self, telemetry: Telemetry) -> None:
        ids = {telemetry.record("e").event_id for _ in range(10)}
        assert len(ids) == 10

    def test_record_stores_event_id_in_session(
        self, telemetry: Telemetry
    ) -> None:
        ev = telemetry.record("e")
        assert ev is not None
        events = telemetry.get_session_events()
        assert ev in events

    def test_record_with_error_status(self, telemetry: Telemetry) -> None:
        ev = telemetry.record(
            "tool.call",
            data={"tool": "x"},
            duration_ms=5.0,
            status="error",
            error="boom",
        )
        assert ev is not None
        assert ev.status == "error"
        assert ev.error == "boom"
        assert ev.duration_ms == 5.0

    def test_record_disabled_returns_none(
        self, tmp_telemetry_path: Path
    ) -> None:
        t = Telemetry(
            output_path=tmp_telemetry_path, enabled=False
        )
        assert t.record("x") is None
        assert t.get_session_events() == []

    def test_record_increments_buffer(
        self, telemetry: Telemetry
    ) -> None:
        assert len(telemetry._buffer) == 0
        telemetry.record("a")
        assert len(telemetry._buffer) == 1
        telemetry.record("b")
        # buffer_size=2 so second record triggers auto-flush
        assert len(telemetry._buffer) == 0
        assert len(telemetry.get_session_events()) == 2

    def test_record_exception_returns_none(
        self, telemetry: Telemetry
    ) -> None:
        class _RaisingLock:
            def __enter__(self):
                raise RuntimeError("lock broken")

            def __exit__(self, *args):
                return False

        with patch.object(telemetry, "_lock", new=_RaisingLock()):
            result = telemetry.record("x")
        assert result is None


# ---------------------------------------------------------------------------
# track() context manager
# ---------------------------------------------------------------------------


class TestTrack:
    """Telemetry.track() sync context manager."""

    def test_track_yields_event_when_enabled(
        self, telemetry: Telemetry
    ) -> None:
        with telemetry.track("phase", model="x") as ev:
            assert ev is not None
            assert ev.event_type == "phase"
            assert ev.data == {"model": "x"}
            assert ev.status == "ok"
            assert ev.error is None
            assert ev.duration_ms == 0.0  # not finalized yet
        assert ev.duration_ms > 0.0  # finalized after exit

    def test_track_measures_duration(self, telemetry: Telemetry) -> None:
        with telemetry.track("slow") as ev:
            time.sleep(0.02)
        assert ev is not None
        assert ev.duration_ms >= 15.0  # generous lower bound

    def test_track_captures_exception(
        self, telemetry: Telemetry
    ) -> None:
        ev = None
        with pytest.raises(RuntimeError, match="boom"):
            with telemetry.track("phase") as ev:
                raise RuntimeError("boom")
        assert ev is not None
        assert ev.status == "error"
        assert ev.error is not None
        assert "RuntimeError" in ev.error
        assert "boom" in ev.error

    def test_track_records_event_in_session(
        self, telemetry: Telemetry
    ) -> None:
        with telemetry.track("phase"):
            pass
        events = telemetry.get_session_events()
        assert len(events) == 1
        assert events[0].event_type == "phase"

    def test_track_yields_none_when_disabled(
        self, tmp_telemetry_path: Path
    ) -> None:
        t = Telemetry(output_path=tmp_telemetry_path, enabled=False)
        with t.track("phase") as ev:
            assert ev is None
        assert t.get_session_events() == []

    def test_track_data_passed_via_kwargs(
        self, telemetry: Telemetry
    ) -> None:
        with telemetry.track(
            "llm.generate", model="qwen2.5:7b", level="moderate"
        ) as ev:
            assert ev is not None
        assert ev.data == {"model": "qwen2.5:7b", "level": "moderate"}

    def test_track_event_id_is_uuid_hex(
        self, telemetry: Telemetry
    ) -> None:
        with telemetry.track("x") as ev:
            assert ev is not None
        assert len(ev.event_id) == 32
        int(ev.event_id, 16)

    def test_track_finalization_failure_does_not_propagate(
        self, telemetry: Telemetry, caplog
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="core.telemetry"):

            class _RaisingLock:
                def __enter__(self):
                    raise RuntimeError("lock down")

                def __exit__(self, *args):
                    return False

            with patch.object(telemetry, "_lock", new=_RaisingLock()):
                with telemetry.track("phase"):
                    pass
        assert "Telemetry track finalization failed" in caplog.text

    def test_track_appends_event_even_on_error(
        self, telemetry: Telemetry
    ) -> None:
        with pytest.raises(ValueError):
            with telemetry.track("phase"):
                raise ValueError("x")
        events = telemetry.get_session_events()
        assert len(events) == 1
        assert events[0].status == "error"


# ---------------------------------------------------------------------------
# JSONL persistence
# ---------------------------------------------------------------------------


class TestFlush:
    """JSONL file writing and flush triggers."""

    def test_record_writes_jsonl_line(
        self, telemetry: Telemetry, tmp_telemetry_path: Path
    ) -> None:
        telemetry.record("a", data={"k": 1})
        telemetry.flush()
        text = tmp_telemetry_path.read_text(encoding="utf-8").strip()
        assert text
        obj = json.loads(text)
        assert obj["event_type"] == "a"
        assert obj["data"] == {"k": 1}

    def test_one_event_per_line(
        self, telemetry: Telemetry, tmp_telemetry_path: Path
    ) -> None:
        for i in range(5):
            telemetry.record(f"e{i}")
        telemetry.flush()
        lines = [
            ln
            for ln in tmp_telemetry_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        assert len(lines) == 5

    def test_buffer_size_auto_flush(
        self, tmp_telemetry_path: Path
    ) -> None:
        t = Telemetry(
            session_id="s", output_path=tmp_telemetry_path, buffer_size=3
        )
        t.record("a")
        t.record("b")
        assert tmp_telemetry_path.exists() is False or tmp_telemetry_path.stat().st_size == 0
        t.record("c")  # third record triggers flush
        text = tmp_telemetry_path.read_text(encoding="utf-8")
        assert text.count("\n") == 3

    def test_flush_empty_buffer_noop(
        self, telemetry: Telemetry, tmp_telemetry_path: Path
    ) -> None:
        telemetry.flush()
        assert not tmp_telemetry_path.exists()

    def test_flush_clears_buffer(
        self, telemetry: Telemetry
    ) -> None:
        telemetry.record("a")
        telemetry.flush()
        assert telemetry._buffer == []

    def test_flush_appends_not_overwrites(
        self, tmp_telemetry_path: Path
    ) -> None:
        t1 = Telemetry(session_id="s1", output_path=tmp_telemetry_path)
        t1.record("first")
        t1.flush()
        t2 = Telemetry(session_id="s2", output_path=tmp_telemetry_path)
        t2.record("second")
        t2.flush()
        text = tmp_telemetry_path.read_text(encoding="utf-8").strip()
        lines = [ln for ln in text.splitlines() if ln]
        assert len(lines) == 2
        assert json.loads(lines[0])["event_type"] == "first"
        assert json.loads(lines[1])["event_type"] == "second"

    def test_jsonl_handles_unicode(
        self, tmp_telemetry_path: Path
    ) -> None:
        t = Telemetry(session_id="s", output_path=tmp_telemetry_path)
        t.record("unicode", data={"msg": "مرحبا 🌍"})
        t.flush()
        text = tmp_telemetry_path.read_text(encoding="utf-8")
        assert "مرحبا 🌍" in text

    def test_flush_failure_logs_warning(
        self, telemetry: Telemetry, caplog
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="core.telemetry"):
            with patch("builtins.open", side_effect=OSError("disk full")):
                telemetry.record("a")
                telemetry.record("b")  # buffer_size=2 -> flush attempted
        assert "Telemetry flush failed" in caplog.text


# ---------------------------------------------------------------------------
# Session events
# ---------------------------------------------------------------------------


class TestGetSessionEvents:
    """get_session_events() lock-safe list copy."""

    def test_empty_session(self, telemetry: Telemetry) -> None:
        assert telemetry.get_session_events() == []

    def test_returns_list_copy(self, telemetry: Telemetry) -> None:
        telemetry.record("a")
        events = telemetry.get_session_events()
        events.append(MagicMock())
        # internal state not mutated
        assert len(telemetry.get_session_events()) == 1

    def test_preserves_order(self, telemetry: Telemetry) -> None:
        for i in range(5):
            telemetry.record(f"e{i}")
        events = telemetry.get_session_events()
        assert [e.event_type for e in events] == ["e0", "e1", "e2", "e3", "e4"]


# ---------------------------------------------------------------------------
# report()
# ---------------------------------------------------------------------------


class TestReport:
    """Telemetry.report() session summary."""

    def test_report_empty_session(
        self, telemetry: Telemetry
    ) -> None:
        r = telemetry.report()
        assert r == {
            "session_id": "test-session",
            "total_events": 0,
            "output_path": str(telemetry.output_path),
        }

    def test_report_total_events(self, telemetry: Telemetry) -> None:
        for _ in range(3):
            telemetry.record("e")
        assert telemetry.report()["total_events"] == 3

    def test_report_by_type_counts(self, telemetry: Telemetry) -> None:
        telemetry.record("llm.generate")
        telemetry.record("llm.generate")
        telemetry.record("tool.call")
        r = telemetry.report()
        assert r["by_type"] == {"llm.generate": 2, "tool.call": 1}

    def test_report_by_status_counts(self, telemetry: Telemetry) -> None:
        telemetry.record("ok1", status="ok")
        telemetry.record("err1", status="error")
        telemetry.record("ok2", status="ok")
        r = telemetry.report()
        assert r["by_status"] == {"ok": 2, "error": 1, "timeout": 0}

    def test_report_avg_duration_per_type(
        self, telemetry: Telemetry
    ) -> None:
        with telemetry.track("slow"):
            time.sleep(0.01)
        with telemetry.track("slow"):
            time.sleep(0.01)
        r = telemetry.report()
        assert "slow" in r["avg_duration_ms"]
        assert r["avg_duration_ms"]["slow"] >= 5.0

    def test_report_errors_collects_first_10(
        self, telemetry: Telemetry
    ) -> None:
        for i in range(15):
            telemetry.record(
                f"e{i}", status="error", error=f"err{i}"
            )
        r = telemetry.report()
        assert len(r["errors"]) == 10
        assert r["errors"][0] == {"type": "e0", "error": "err0"}

    def test_report_errors_excludes_ones_without_error_message(
        self, telemetry: Telemetry
    ) -> None:
        telemetry.record("e", status="error", error=None)
        r = telemetry.report()
        assert r["errors"] == []

    def test_report_includes_session_id_and_path(
        self, telemetry: Telemetry
    ) -> None:
        r = telemetry.report()
        assert r["session_id"] == "test-session"
        assert r["output_path"] == str(telemetry.output_path)


# ---------------------------------------------------------------------------
# reset() and close()
# ---------------------------------------------------------------------------


class TestResetAndClose:
    """reset() and close() lifecycle methods."""

    def test_reset_clears_events_and_buffer(
        self, telemetry: Telemetry
    ) -> None:
        telemetry.record("a")
        telemetry.record("b")
        assert len(telemetry.get_session_events()) == 2
        telemetry.reset()
        assert telemetry.get_session_events() == []
        assert telemetry._buffer == []

    def test_close_flushes_buffer(
        self, telemetry: Telemetry, tmp_telemetry_path: Path
    ) -> None:
        telemetry.record("a")
        # buffer not yet flushed
        telemetry.close()
        assert tmp_telemetry_path.exists()
        text = tmp_telemetry_path.read_text(encoding="utf-8").strip()
        assert json.loads(text)["event_type"] == "a"

    def test_close_does_not_clear_events(
        self, telemetry: Telemetry
    ) -> None:
        telemetry.record("a")
        telemetry.close()
        # events remain after close (only buffer is flushed)
        assert len(telemetry.get_session_events()) == 1


# ---------------------------------------------------------------------------
# Error resilience
# ---------------------------------------------------------------------------


class TestErrorResilience:
    """Telemetry must never break the calling code."""

    def test_ensure_path_failure_disables_telemetry(
        self, tmp_telemetry_path: Path, caplog
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="core.telemetry"):
            with patch(
                "pathlib.Path.mkdir",
                side_effect=OSError("permission denied"),
            ):
                t = Telemetry(output_path=tmp_telemetry_path)
        assert t.enabled is False
        assert "cannot create" in caplog.text

    def test_disabled_record_logs_nothing(
        self, tmp_telemetry_path: Path, caplog
    ) -> None:
        t = Telemetry(output_path=tmp_telemetry_path, enabled=False)
        with caplog.at_level(logging.WARNING, logger="core.telemetry"):
            result = t.record("x")
        assert result is None
        # no warnings expected for disabled telemetry
        assert "Telemetry record failed" not in caplog.text

    def test_disabled_track_does_not_record(
        self, tmp_telemetry_path: Path
    ) -> None:
        t = Telemetry(output_path=tmp_telemetry_path, enabled=False)
        with t.track("x") as ev:
            assert ev is None
        assert t.get_session_events() == []

    def test_module_constants_exist(self) -> None:
        # sanity: module exposes expected constants
        assert isinstance(TELEMETRY_DIR, Path)
        assert isinstance(TELEMETRY_FILE, Path)
        assert TELEMETRY_FILE.parent == TELEMETRY_DIR


# ---------------------------------------------------------------------------
# Session ID
# ---------------------------------------------------------------------------


class TestSessionId:
    """_new_session_id() format and uniqueness."""

    def test_format(self) -> None:
        sid = Telemetry._new_session_id()
        assert sid.startswith("session-")
        # session-<int_ts>-<8hex>
        parts = sid.split("-")
        assert len(parts) == 3
        assert parts[0] == "session"
        int(parts[1])  # timestamp
        assert len(parts[2]) == 8
        int(parts[2], 16)

    def test_unique(self) -> None:
        ids = {Telemetry._new_session_id() for _ in range(100)}
        assert len(ids) == 100


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Concurrent record() must not lose events or corrupt the buffer."""

    def test_concurrent_record(self, telemetry: Telemetry) -> None:
        n_threads = 10
        n_per_thread = 50
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                for _ in range(n_per_thread):
                    telemetry.record("concurrent")
            except BaseException as e:  # noqa: BLE001
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []
        # All events captured (no losses under the lock)
        assert len(telemetry.get_session_events()) == n_threads * n_per_thread
