"""Core notifications module for AI Agent"""
import json
import uuid
from datetime import datetime
from typing import List, Optional, Dict
from pathlib import Path


class Notification:
    def __init__(self, user_id: str, title: str, message: str, ntype: str = "info"):
        self.id = str(uuid.uuid4())[:8]
        self.user_id = user_id
        self.title = title
        self.message = message
        self.type = ntype
        self.read = False
        self.created_at = datetime.now().isoformat()

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "message": self.message,
            "type": self.type,
            "read": self.read,
            "created_at": self.created_at
        }


class NotificationManager:
    def __init__(self, storage_path: str = "notifications.json"):
        self.storage_path = Path(storage_path)
        self.notifications: List[Notification] = []
        self._loaded = False
        self._load()

    def _load(self):
        if self._loaded:
            return
        if self.storage_path.exists():
            try:
                data = json.loads(self.storage_path.read_text(encoding="utf-8"))
                for item in data:
                    notif = Notification(
                        item["user_id"],
                        item["title"],
                        item["message"],
                        item.get("type", "info")
                    )
                    notif.id = item.get("id", notif.id)
                    notif.read = item.get("read", False)
                    notif.created_at = item.get("created_at", notif.created_at)
                    self.notifications.append(notif)
            except Exception:
                self.notifications = []
        self._loaded = True

    def _save(self):
        data = [n.to_dict() for n in self.notifications]
        self.storage_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    def add_notification(self, user_id: str, title: str, message: str, ntype: str = "info") -> bool:
        notif = Notification(user_id, title, message, ntype)
        self.notifications.append(notif)
        self._save()
        return True

    def get_notifications(self, user_id: str, unread_only: bool = False) -> List[Dict]:
        result = []
        for n in self.notifications:
            if n.user_id == user_id:
                if unread_only and n.read:
                    continue
                result.append(n.to_dict())
        return result

    def mark_read(self, notification_id: str) -> bool:
        for n in self.notifications:
            if n.id == notification_id:
                n.read = True
                self._save()
                return True
        return False

    def get_unread_count(self, user_id: str) -> int:
        return len([n for n in self.notifications if n.user_id == user_id and not n.read])

    def delete_notification(self, notification_id: str) -> bool:
        for i, n in enumerate(self.notifications):
            if n.id == notification_id:
                self.notifications.pop(i)
                self._save()
                return True
        return False

    def clear_read(self, user_id: str) -> int:
        before = len(self.notifications)
        self.notifications = [n for n in self.notifications if not (n.user_id == user_id and n.read)]
        self._save()
        return before - len(self.notifications)
