"""Task Scheduling - periodic and scheduled tasks"""

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
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable] = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    self.tasks = json.load(f)
            except Exception:
                self.tasks = []
        self._loaded = True

    def set_callback(self, callback: Callable):
        self._callback = callback

    def add_task(self, name: str, prompt: str, schedule: str, task_type: str = "interval") -> dict:
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
        }
        self.tasks.append(task)
        self._save()
        return task

    def remove_task(self, task_id: str) -> bool:
        for i, t in enumerate(self.tasks):
            if t["id"] == task_id:
                self.tasks.pop(i)
                self._save()
                return True
        return False

    def list_tasks(self) -> list[dict]:
        self.load()
        return self.tasks

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _run_loop(self):
        while self._running:
            self.load()
            now = datetime.now()
            for task in self.tasks:
                if not task.get("enabled", True):
                    continue
                next_run = task.get("next_run", "")
                if next_run and now >= datetime.fromisoformat(next_run):
                    try:
                        if self._callback:
                            self._callback(task["name"], task["prompt"])
                    except Exception:
                        pass
                    finally:
                        task["last_run"] = now.isoformat()
                        task["next_run"] = self._calculate_next_run(
                            task["schedule"], task.get("task_type", "interval")
                        )
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
        return (now + timedelta(hours=1)).isoformat()

    def _save(self):
        try:
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(self.tasks, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
