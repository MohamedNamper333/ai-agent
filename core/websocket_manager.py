"""WebSocket connection manager — real-time bidirectional communication."""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict
from typing import Any, Optional

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manage active WebSocket connections with room support.

    Provides real-time bidirectional messaging between clients and the AI agent,
    including progress events during tool execution and multi-user room support.
    """

    def __init__(self):
        """Initialize the connection manager."""
        # conn_id → WebSocket
        self._active: dict[str, WebSocket] = {}
        # room_id → set of conn_ids
        self._rooms: dict[str, set[str]] = defaultdict(set)
        # conn_id → metadata
        self._meta: dict[str, dict] = {}
        self._total_connections = 0
        self._total_messages = 0

    async def connect(self, ws: WebSocket, conn_id: Optional[str] = None,
                      room: str = "default") -> str:
        """Accept and register a new WebSocket connection.

        Returns:
            The assigned connection ID.
        """
        await ws.accept()
        conn_id = conn_id or str(uuid.uuid4())[:8]
        self._active[conn_id] = ws
        self._rooms[room].add(conn_id)
        self._meta[conn_id] = {
            "conn_id": conn_id,
            "room": room,
            "connected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "messages_sent": 0,
            "messages_received": 0,
        }
        self._total_connections += 1
        logger.info("WebSocket connected: %s (room=%s, total=%d)",
                    conn_id, room, len(self._active))
        await self.send_json(conn_id, {
            "type": "connected",
            "conn_id": conn_id,
            "room": room,
            "ts": time.time(),
        })
        return conn_id

    def disconnect(self, conn_id: str) -> None:
        """Remove a disconnected client from all rooms."""
        if conn_id in self._active:
            del self._active[conn_id]
        meta = self._meta.pop(conn_id, {})
        room = meta.get("room", "default")
        self._rooms[room].discard(conn_id)
        if not self._rooms[room]:
            del self._rooms[room]
        logger.info("WebSocket disconnected: %s", conn_id)

    async def send_json(self, conn_id: str, data: dict) -> bool:
        """Send a JSON message to a specific connection.

        Returns:
            True if sent successfully, False otherwise.
        """
        ws = self._active.get(conn_id)
        if not ws:
            return False
        try:
            await ws.send_json(data)
            self._total_messages += 1
            if conn_id in self._meta:
                self._meta[conn_id]["messages_sent"] += 1
            return True
        except Exception as exc:
            logger.warning("WebSocket send failed for %s: %s", conn_id, exc)
            self.disconnect(conn_id)
            return False

    async def broadcast_room(self, room: str, data: dict) -> int:
        """Broadcast a message to all connections in a room.

        Returns:
            Number of clients successfully reached.
        """
        conn_ids = list(self._rooms.get(room, set()))
        results = await asyncio.gather(
            *[self.send_json(cid, data) for cid in conn_ids],
            return_exceptions=True,
        )
        return sum(1 for r in results if r is True)

    async def send_progress(self, conn_id: str, tool_name: str,
                            status: str, detail: Optional[str] = None) -> None:
        """Send a tool execution progress event to a client."""
        await self.send_json(conn_id, {
            "type": "tool_progress",
            "tool": tool_name,
            "status": status,  # "running" | "done" | "error"
            "detail": detail,
            "ts": time.time(),
        })

    async def send_chunk(self, conn_id: str, text: str) -> None:
        """Send a streaming text chunk to a client."""
        await self.send_json(conn_id, {
            "type": "chunk",
            "text": text,
            "ts": time.time(),
        })

    async def send_error(self, conn_id: str, message: str, code: int = 400) -> None:
        """Send an error message to a client."""
        await self.send_json(conn_id, {
            "type": "error",
            "message": message,
            "code": code,
            "ts": time.time(),
        })

    async def send_done(self, conn_id: str, full_text: str = "") -> None:
        """Send a completion signal to a client."""
        await self.send_json(conn_id, {
            "type": "done",
            "full_text": full_text,
            "ts": time.time(),
        })

    def get_stats(self) -> dict:
        """Return connection manager statistics."""
        return {
            "active_connections": len(self._active),
            "active_rooms": len(self._rooms),
            "total_connections_ever": self._total_connections,
            "total_messages_sent": self._total_messages,
            "rooms": {r: len(c) for r, c in self._rooms.items()},
        }


# Singleton manager
_manager: Optional[ConnectionManager] = None


def get_ws_manager() -> ConnectionManager:
    """Return the global WebSocket connection manager singleton."""
    global _manager
    if _manager is None:
        _manager = ConnectionManager()
    return _manager
