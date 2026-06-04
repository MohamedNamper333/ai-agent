"""Lightweight telemetry for the agent runtime.

Design goals
------------
* Zero-config — works without any external service.
* Thread-safe — multiple threads may record events concurrently.
* Bounded — never blow up disk; auto-rotate when the JSONL gets large.
* Inspectable — :meth:`Telemetry.report` returns a quick stats summary.

The module deliberately avoids async. Callers wrap the ``@track``
context-manager in their async functions if they want; we capture wall
clock only, not coroutine depth.

W1 success criteria
-------------------
* 10 unit tests covering write/read/rotate/stats/thread-safety.
* ``logs/telemetry.jsonl`` is created on first event.
* ``report()`` is deterministic for a given event stream.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional


DEFAULT_LOG_DIR = Path("logs")
DEFAULT_LOG_FILE = "telemetry.jsonl"
MAX_LOG_BYTES = 5 * 1024 * 1024  # 5 MiB; rotate beyond this
BACKUP_KEEP = 3  # keep last 3 rotated files


@dataclass
class AgentEvent:
    """A single recorded event.

    ``event_id`` is a UUID4 assigned at construction. ``ts`` is an ISO-8601
    UTC timestamp. ``duration_ms`` is set by the context-manager when an
    event finishes; it is 0 for instantaneous events.
    """

    name: str
    status: str = "ok"
    duration_ms: int = 0
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    ts: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    data: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


class Telemetry:
    """File-backed event recorder.

    Use :meth:`track` to wrap a block of code, or :meth:`event` to record
    a point-in-time observation. :meth:`report` aggregates the in-memory
    ring buffer into a dict suitable for logging.

    Parameters
    ----------
    log_dir : str | Path
        Directory that will contain ``telemetry.jsonl``. Created on
        first write.
    max_events : int
        Size of the in-memory ring buffer used for ``report()``. Older
        events fall off the back.
    enabled : bool
        When False, :meth:`track` / :meth:`event` become no-ops. Lets
        callers disable telemetry in tests by setting the env var
        ``AGENT_TELEMETRY=0``.
    """

    def __init__(
        self,
        log_dir: str | Path = DEFAULT_LOG_DIR,
        max_events: int = 1000,
        enabled: bool = True,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_path = self.log_dir / DEFAULT_LOG_FILE
        self.max_events = max(1, int(max_events))
        self.enabled = bool(enabled)
        self._buffer: list[AgentEvent] = []
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._counters: Counter[str] = Counter()

    # -- public surface -------------------------------------------------------

    @contextlib.contextmanager
    def track(self, name: str, **data: Any) -> Iterator[AgentEvent]:
        """Wrap a block; record one ``ok`` or ``error`` event on exit."""
        if not self.enabled:
            yield AgentEvent(name="noop")
            return
        event = AgentEvent(name=name, data=dict(data))
        # perf_counter() — not monotonic() — is always backed by the highest
        # resolution timer available on the platform (QueryPerformanceCounter
        # on Windows, clock_gettime(CLOCK_MONOTONIC_RAW) on Linux). monotonic()
        # may fall back to GetTickCount64 (~15.6ms) on some Windows builds,
        # which would make a 2ms sleep read as 0ms and trip the unit tests.
        start = time.perf_counter()
        try:
            yield event
            event.status = "ok"
        except BaseException as exc:  # noqa: BLE001 - re-raised after record
            event.status = "error"
            event.error = f"{type(exc).__name__}: {exc}"
            event.data["error"] = event.error
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            # Round to nearest ms, but if the elapsed time is positive yet
            # still rounds to 0 (because the clock tick is sub-ms), surface
            # it as 1ms so callers/tests never see a phantom 0-duration event
            # when real work happened.
            rounded = int(round(elapsed_ms))
            if rounded == 0 and elapsed_ms > 0:
                rounded = 1
            event.duration_ms = rounded
            self._record(event)

    def event(self, name: str, **data: Any) -> AgentEvent:
        """Record a point-in-time event (no duration)."""
        payload = dict(data)
        duration = payload.pop("duration_ms", 0) or 0
        ev = AgentEvent(
            name=name,
            duration_ms=int(duration),
            data=payload,
        )
        if self.enabled:
            self._record(ev)
        return ev

    def report(self) -> dict[str, Any]:
        """Aggregate summary of the in-memory ring buffer."""
        with self._lock:
            events = list(self._buffer)
        counters: Counter[str] = Counter()
        durations: list[int] = []
        errors = 0
        for ev in events:
            counters[ev.name] += 1
            durations.append(ev.duration_ms)
            if ev.status == "error":
                errors += 1
        durations.sort()
        n = len(durations) or 1
        return {
            "total_events": len(events),
            "by_name": dict(counters),
            "errors": errors,
            "duration_ms": {
                "min": durations[0] if durations else 0,
                "max": durations[-1] if durations else 0,
                "avg": sum(durations) // n if durations else 0,
                "p50": _percentile(durations, 50),
                "p95": _percentile(durations, 95),
            },
        }

    def recent(self, limit: int = 20) -> list[AgentEvent]:
        """Return the most recent ``limit`` events (in order)."""
        with self._lock:
            return list(self._buffer[-limit:])

    # -- internals -----------------------------------------------------------

    def _record(self, event: AgentEvent) -> None:
        with self._lock:
            self._buffer.append(event)
            if len(self._buffer) > self.max_events:
                drop = len(self._buffer) - self.max_events
                del self._buffer[:drop]
            self._counters[event.name] += 1
        if self.enabled:
            self._write(event)

    def _write(self, event: AgentEvent) -> None:
        with self._write_lock:
            try:
                self._maybe_rotate()
                self.log_dir.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as fh:
                    fh.write(event.to_json())
                    fh.write("\n")
            except OSError:
                # Disk-full, permission, etc. — telemetry must not crash
                # the host process. Swallow.
                pass

    def _maybe_rotate(self) -> None:
        if not self.log_path.exists():
            return
        try:
            if self.log_path.stat().st_size < MAX_LOG_BYTES:
                return
        except OSError:
            return
        # Rotate: telemetry.jsonl → telemetry.1.jsonl → telemetry.2.jsonl
        for i in range(BACKUP_KEEP - 1, 0, -1):
            src = self.log_dir / f"telemetry.{i}.jsonl"
            dst = self.log_dir / f"telemetry.{i + 1}.jsonl"
            if src.exists():
                try:
                    if dst.exists():
                        dst.unlink()
                    src.rename(dst)
                except OSError:
                    pass
        first = self.log_dir / "telemetry.1.jsonl"
        try:
            if first.exists():
                first.unlink()
            self.log_path.rename(first)
        except OSError:
            pass


def _percentile(sorted_values: list[int], p: int) -> int:
    if not sorted_values:
        return 0
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = max(0, min(len(sorted_values) - 1, int(round((p / 100.0) * (len(sorted_values) - 1)))))
    return sorted_values[k]


def is_telemetry_enabled() -> bool:
    """Read the ``AGENT_TELEMETRY`` env var. Defaults to enabled."""
    raw = os.environ.get("AGENT_TELEMETRY", "1").strip().lower()
    return raw not in ("0", "false", "no", "off", "disabled")


def default_telemetry() -> Telemetry:
    """Build a ``Telemetry`` instance respecting the env flag."""
    return Telemetry(enabled=is_telemetry_enabled())


__all__ = [
    "AgentEvent",
    "Telemetry",
    "default_telemetry",
    "is_telemetry_enabled",
    "DEFAULT_LOG_DIR",
    "DEFAULT_LOG_FILE",
    "MAX_LOG_BYTES",
    "BACKUP_KEEP",
]
