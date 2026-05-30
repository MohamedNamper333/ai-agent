import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import config


class Message:
    def __init__(self, role: str, content: str, timestamp: Optional[str] = None):
        self.role = role
        self.content = content
        self.timestamp = timestamp or datetime.now().isoformat()

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", ""),
        )


class ConversationMemory:
    def __init__(self, max_tokens: int = 6000, db_path: str = ""):
        self.max_tokens = max_tokens
        self.db_path = db_path or config.DB_PATH
        self.conversations: dict[str, list[Message]] = {}
        self.current_id: str = ""
        self._loaded = False

    def _get_path(self) -> str:
        return str(Path(config.BASE_DIR) / self.db_path)

    def new_conversation(self, conversation_id: str = "") -> str:
        cid = conversation_id or datetime.now().strftime("conv_%Y%m%d_%H%M%S")
        self.conversations[cid] = []
        self.current_id = cid
        return cid

    def add_message(self, role: str, content: str) -> None:
        if not self.current_id or self.current_id not in self.conversations:
            self.new_conversation()
        self.conversations[self.current_id].append(Message(role, content))
        self._save()

    def get_history(self, conversation_id: str = "") -> list[dict]:
        cid = conversation_id or self.current_id
        msgs = self.conversations.get(cid, [])
        return [m.to_dict() for m in msgs]

    def get_trimmed_history(self, max_tokens: int = 0) -> list[dict]:
        mt = max_tokens or self.max_tokens
        msgs = self.conversations.get(self.current_id, [])
        total = sum(len(m.content) // 3 for m in msgs)
        while msgs and total > mt:
            removed = msgs.pop(0)
            total -= len(removed.content) // 3
        return [m.to_dict() for m in msgs]

    def format_for_llm(self, system_prompt: str, include_system: bool = True) -> str:
        parts = []
        if include_system and system_prompt:
            parts.append(f"<|system|>\n{system_prompt}\n")
        for m in self.get_trimmed_history():
            role = "user" if m["role"] == "user" else "assistant"
            parts.append(f"<|{role}|>\n{m['content']}\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)

    def list_conversations(self) -> list[str]:
        return list(self.conversations.keys())

    def delete_conversation(self, conversation_id: str) -> bool:
        return self.conversations.pop(conversation_id, None) is not None

    def _save(self) -> None:
        try:
            data = {
                cid: [m.to_dict() for m in msgs]
                for cid, msgs in self.conversations.items()
            }
            path = self._get_path()
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[memory] Warning: save failed: {e}")

    def load(self) -> None:
        if self._loaded:
            return
        path = self._get_path()
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for cid, msgs in data.items():
                    self.conversations[cid] = [
                        Message.from_dict(m) for m in msgs
                    ]
                if self.conversations:
                    self.current_id = list(self.conversations.keys())[-1]
            except Exception as e:
                print(f"[memory] Warning: load failed: {e}")
        self._loaded = True

    def clear(self) -> None:
        self.conversations = {}
        self.current_id = ""
        path = self._get_path()
        if os.path.exists(path):
            os.remove(path)
