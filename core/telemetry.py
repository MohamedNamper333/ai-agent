"""Agent telemetry — captures events for monitoring, debugging, and (Phase 2)
self-improvement.

Writes JSONL events to logs/telemetry.jsonl.  Best-effort: telemetry
failures NEVER break the agent.

Provides:
  - AgentEvent dataclass for structured events
  - Telemetry class with sync + async context managers
  - @track decorator / contextmanager for capturing event phases
  - .report() method for session summary
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Iterator, Optional


logger = logging.getLogger(__name__)

try:
    from config.env_loader import BASE_DIR
except ImportError:
    BASE_DIR = Path(__file__).resolve().parent.parent


TELEMETRY_DIR = BASE_DIR / "logs"
TELEMETRY_FILE = TELEMETRY_DIR / "telemetry.jsonl"


@dataclass
class AgentEvent:
    """A single telemetry event."""

    event_id: str
    event_type: str
    timestamp: float
    session_id: str
    duration_ms: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"
    error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Telemetry:
    """Capture and persist agent events. Best-effort, thread-safe."""

    def __init__(
        self,
        session_id: Optional[str] = None,
        output_path: Optional[Path] = None,
        buffer_size: int = 10,
        enabled: bool = True,
    ):
        self.session_id = session_id or self._new_session_id()
        self.output_path = output_path or TELEMETRY_FILE
        self.buffer_size = max(1, int(buffer_size))
        self.enabled = enabled
        self._events: list[AgentEvent] = []
        self._buffer: list[AgentEvent] = []
        self._lock = Lock()
        if self.enabled:
            self._ensure_path()

    @staticmethod
    def _new_session_id() -> str:
        return f"session-{int(time.time())}-{uuid.uuid4().hex[:8]}"

    def _ensure_path(self) -> None:
        try:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(
                "Telemetry: cannot create %s: %s", self.output_path.parent, e
            )
            self.enabled = False

    def record(
        self,
        event_type: str,
        data: Optional[dict] = None,
        duration_ms: float = 0.0,
        status: str = "ok",
        error: Optional[str] = None,
    ) -> Optional[AgentEvent]:
        """Record an event synchronously. Returns the event or None."""
        if not self.enabled:
            return None
        try:
            event = AgentEvent(
                event_id=uuid.uuid4().hex,
                event_type=event_type,
                timestamp=time.time(),
                session_id=self.session_id,
                duration_ms=duration_ms,
                data=data or {},
                status=status,
                error=error,
            )
            with self._lock:
                self._events.append(event)
                self._buffer.append(event)
                if len(self._buffer) >= self.buffer_size:
                    self._flush()
            return event
        except Exception as e:
            logger.warning("Telemetry record failed: %s", e)
            return None

    @contextmanager
    def track(self, event_type: str, **data: Any) -> Iterator[Optional[AgentEvent]]:
        """Context manager: measures duration_ms automatically.

        Usage:
            with telemetry.track("llm.generate", model="qwen2.5"):
                result = llm.generate(prompt)
        """
        if not self.enabled:
            yield None
            return
        event = AgentEvent(
            event_id=uuid.uuid4().hex,
            event_type=event_type,
            timestamp=time.time(),
            session_id=self.session_id,
            data=dict(data),
        )
        start = time.perf_counter()
        try:
            yield event
            event.status = "ok"
        except Exception as e:
            event.status = "error"
            event.error = f"{type(e).__name__}: {e}"
            raise
        finally:
            try:
                event.duration_ms = (time.perf_counter() - start) * 1000.0
                with self._lock:
                    self._events.append(event)
                    self._buffer.append(event)
                    if len(self._buffer) >= self.buffer_size:
                        self._flush()
            except Exception as e:
                logger.warning("Telemetry track finalization failed: %s", e)

    def _flush(self) -> None:
        if not self._buffer:
            return
        try:
            with open(self.output_path, "a", encoding="utf-8") as f:
                for event in self._buffer:
                    f.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")
            self._buffer.clear()
        except Exception as e:
            logger.warning("Telemetry flush failed: %s", e)

    def flush(self) -> None:
        with self._lock:
            self._flush()

    def get_session_events(self) -> list[AgentEvent]:
        with self._lock:
            return list(self._events)

    def report(self) -> dict[str, Any]:
        """Generate a session summary report."""
        with self._lock:
            events = list(self._events)
        if not events:
            return {
                "session_id": self.session_id,
                "total_events": 0,
                "output_path": str(self.output_path),
            }

        by_type: dict[str, int] = {}
        by_status: dict[str, int] = {"ok": 0, "error": 0, "timeout": 0}
        durations_by_type: dict[str, list[float]] = {}
        errors: list[dict[str, Any]] = []

        for ev in events:
            by_type[ev.event_type] = by_type.get(ev.event_type, 0) + 1
            by_status[ev.status] = by_status.get(ev.status, 0) + 1
            durations_by_type.setdefault(ev.event_type, []).append(ev.duration_ms)
            if ev.status == "error" and ev.error:
                errors.append({"type": ev.event_type, "error": ev.error})

        avg_durations = {
            t: round(sum(ds) / len(ds), 2)
            for t, ds in durations_by_type.items()
            if ds
        }

        return {
            "session_id": self.session_id,
            "total_events": len(events),
            "by_type": by_type,
            "by_status": by_status,
            "avg_duration_ms": avg_durations,
            "errors": errors[:10],
            "output_path": str(self.output_path),
        }

    def reset(self) -> None:
        with self._lock:
            self._events.clear()
            self._buffer.clear()

    def close(self) -> None:
        self.flush()
