"""Task Scheduling - periodic, daily, and one-time tasks"""

import json
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Callable

import config


class TaskScheduler:
    def __init__(self, db_path: str = ""):
        self.db_path = db_path or str(Path(config.BASE_DIR) / "scheduled_tasks.json")
        self.tasks: list[dict] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable] = None
        self._loaded = False
        self._history: list[dict] = []

    def load(self):
        """Load data from storage."""
        if self._loaded:
            return
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.tasks = data.get("tasks", [])
                        self._history = data.get("history", [])
                    else:
                        self.tasks = data
            except Exception:
                self.tasks = []
        self._loaded = True

    def set_callback(self, callback: Callable):
        """Set the function to invoke when a scheduled task fires."""
        self._callback = callback

    def add_task(self, name: str, prompt: str, schedule: str, task_type: str = "interval") -> dict:
        """Create and persist a new scheduled task. Return the task dict."""
        if task_type not in ("interval", "daily", "once"):
            return {"error": "task_type must be 'interval', 'daily', or 'once'"}

        task = {
            "id": f"task_{int(time.time())}_{len(self.tasks)}",
            "name": name,
            "prompt": prompt,
            "schedule": schedule,
            "task_type": task_type,
            "enabled": True,
            "last_run": "",
            "next_run": self._calculate_next_run(schedule, task_type),
            "created": datetime.now().isoformat(),
            "run_count": 0,
        }

        with self._lock:
            self.tasks.append(task)
        self._save()
        return task

    def remove_task(self, task_id: str) -> bool:
        """Delete the task with the given ID. Return True if found."""
        with self._lock:
            for i, t in enumerate(self.tasks):
                if t["id"] == task_id:
                    self.tasks.pop(i)
                    self._save()
                    return True
        return False

    def enable_task(self, task_id: str) -> bool:
        """Re-enable a previously disabled task. Return True on success."""
        with self._lock:
            for t in self.tasks:
                if t["id"] == task_id:
                    t["enabled"] = True
                    self._save()
                    return True
        return False

    def disable_task(self, task_id: str) -> bool:
        """Prevent a task from firing without deleting it. Return True on success."""
        with self._lock:
            for t in self.tasks:
                if t["id"] == task_id:
                    t["enabled"] = False
                    self._save()
                    return True
        return False

    def list_tasks(self) -> list[dict]:
        """Return a copy of all registered tasks."""
        self.load()
        with self._lock:
            return list(self.tasks)

    def get_task(self, task_id: str) -> Optional[dict]:
        """Return a copy of the task with the given ID, or None."""
        self.load()
        with self._lock:
            for t in self.tasks:
                if t["id"] == task_id:
                    return dict(t)
        return None

    def get_history(self, limit: int = 20) -> list[dict]:
        """Return the last N execution history entries."""
        return self._history[-limit:]

    def start(self):
        """Start the background scheduling thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the background scheduling thread."""
        self._running = False

    def _run_loop(self):
        while self._running:
            self.load()
            now = datetime.now()
            with self._lock:
                for task in self.tasks:
                    if not task.get("enabled", True):
                        continue
                    next_run = task.get("next_run", "")
                    if next_run and now >= datetime.fromisoformat(next_run):
                        try:
                            if self._callback:
                                result = self._callback(task["name"], task["prompt"])
                                self._history.append({
                                    "task_id": task["id"],
                                    "task_name": task["name"],
                                    "prompt": task["prompt"],
                                    "result": str(result)[:500] if result else "",
                                    "timestamp": now.isoformat(),
                                    "status": "success",
                                })
                        except Exception as e:
                            self._history.append({
                                "task_id": task["id"],
                                "task_name": task["name"],
                                "error": str(e),
                                "timestamp": now.isoformat(),
                                "status": "failed",
                            })
                        finally:
                            task["last_run"] = now.isoformat()
                            task["run_count"] = task.get("run_count", 0) + 1

                            if task.get("task_type") == "once":
                                task["enabled"] = False
                                task["next_run"] = ""
                            else:
                                task["next_run"] = self._calculate_next_run(
                                    task["schedule"], task.get("task_type", "interval")
                                )

                            if len(self._history) > 100:
                                self._history = self._history[-100:]

                            self._save()
            time.sleep(15)

    def _calculate_next_run(self, schedule: str, task_type: str) -> str:
        now = datetime.now()
        if task_type == "interval":
            try:
                minutes = int(schedule)
                return (now + timedelta(minutes=minutes)).isoformat()
            except ValueError:
                return (now + timedelta(hours=1)).isoformat()
        elif task_type == "daily":
            try:
                hour, minute = schedule.split(":")
                next_run = now.replace(hour=int(hour), minute=int(minute), second=0)
                if next_run <= now:
                    next_run += timedelta(days=1)
                return next_run.isoformat()
            except Exception:
                return (now + timedelta(hours=24)).isoformat()
        elif task_type == "once":
            return (now + timedelta(seconds=5)).isoformat()
        return (now + timedelta(hours=1)).isoformat()

    def _save(self):
        try:
            data = {
                "tasks": self.tasks,
                "history": self._history,
            }
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
