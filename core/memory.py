"""Conversation Memory - with smart trimming and search"""

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

    def get_tokens_estimate(self) -> int:
        return max(1, len(self.content) // 4)


class ConversationMemory:
    def __init__(self, max_tokens: int = 6000, db_path: str = ""):
        self.max_tokens = max_tokens
        self.db_path = db_path or config.DB_PATH
        self.conversations: dict[str, list[Message]] = {}
        self.current_id: str = ""
        self._loaded = False
        self._dirty = False
        self._last_save_count = 0

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
        self._dirty = True
        current_count = sum(len(msgs) for msgs in self.conversations.values())
        if current_count - self._last_save_count >= 5 or not self._dirty:
            self._save()
            self._last_save_count = current_count
            self._dirty = False

    def get_history(self, conversation_id: str = "") -> list[dict]:
        cid = conversation_id or self.current_id
        msgs = self.conversations.get(cid, [])
        return [m.to_dict() for m in msgs]

    def get_trimmed_history(self, max_tokens: int = 0) -> list[dict]:
        mt = max_tokens or self.max_tokens
        msgs = list(self.conversations.get(self.current_id, []))
        total = sum(m.get_tokens_estimate() for m in msgs)
        while msgs and total > mt:
            removed = msgs.pop(0)
            total -= removed.get_tokens_estimate()
        return [m.to_dict() for m in msgs]

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        query_lower = query.lower()
        results = []

        for cid, msgs in self.conversations.items():
            for i, msg in enumerate(msgs):
                score = 0
                content_lower = msg.content.lower()

                if query_lower in content_lower:
                    score += 5

                query_words = set(query_lower.split())
                content_words = set(content_lower.split())
                overlap = len(query_words & content_words)
                score += overlap * 0.3

                if score > 0:
                    results.append({
                        "conversation_id": cid,
                        "message_index": i,
                        "role": msg.role,
                        "content": msg.content[:200],
                        "timestamp": msg.timestamp,
                        "score": score,
                    })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

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

    def get_conversation_summary(self, conversation_id: str) -> str:
        msgs = self.conversations.get(conversation_id, [])
        if not msgs:
            return "Empty conversation"

        user_msgs = [m for m in msgs if m.role == "user"]
        assistant_msgs = [m for m in msgs if m.role == "assistant"]

        return (
            f"Conversation: {conversation_id}\n"
            f"Messages: {len(msgs)} ({len(user_msgs)} user, {len(assistant_msgs)} assistant)\n"
            f"First message: {msgs[0].timestamp}\n"
            f"Last message: {msgs[-1].timestamp}\n"
            f"Topics: {', '.join(m.content[:50] for m in user_msgs[:3])}"
        )

    def get_stats(self) -> dict:
        total_msgs = sum(len(msgs) for msgs in self.conversations.values())
        total_tokens = sum(
            m.get_tokens_estimate()
            for msgs in self.conversations.values()
            for m in msgs
        )
        return {
            "conversations": len(self.conversations),
            "total_messages": total_msgs,
            "estimated_tokens": total_tokens,
            "current_conversation": self.current_id,
        }

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
        self._dirty = False
        self._last_save_count = sum(len(msgs) for msgs in self.conversations.values())

    def save_if_dirty(self) -> None:
        if self._dirty:
            self._save()
            self._dirty = False
            self._last_save_count = sum(len(msgs) for msgs in self.conversations.values())

    def clear(self) -> None:
        self.conversations = {}
        self.current_id = ""
        path = self._get_path()
        if os.path.exists(path):
            os.remove(path)
