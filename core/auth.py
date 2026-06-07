"""Authentication & Authorization System"""
import os
import time
import hashlib
import secrets
from typing import Optional
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path


class UserRole(Enum):
    ANONYMOUS = "anonymous"
    BASIC = "basic"
    PREMIUM = "premium"
    ADMIN = "admin"


@dataclass
class User:
    user_id: str
    username: str
    role: UserRole
    api_key: str = ""
    created_at: float = 0.0
    last_login: float = 0.0
    is_active: bool = True

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "role": self.role.value,
            "api_key": self.api_key,
            "created_at": self.created_at,
            "last_login": self.last_login,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "User":
        return cls(
            user_id=data["user_id"],
            username=data["username"],
            role=UserRole(data.get("role", "basic")),
            api_key=data.get("api_key", ""),
            created_at=data.get("created_at", 0.0),
            last_login=data.get("last_login", 0.0),
            is_active=data.get("is_active", True),
        )


class AuthManager:
    def __init__(self, db_path: str = ""):
        self.db_path = db_path or str(Path(os.getcwd()) / "users.json")
        self._users: dict[str, User] = {}
        self._api_keys: dict[str, str] = {}
        self._session_tokens: dict[str, dict] = {}
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for user_data in data.get("users", []):
                    user = User.from_dict(user_data)
                    self._users[user.user_id] = user
                    if user.api_key:
                        self._api_keys[user.api_key] = user.user_id
            except Exception as e:
                print(f"[auth] Warning: load failed: {e}")
        self._loaded = True

    def _save(self):
        try:
            data = {
                "users": [user.to_dict() for user in self._users.values()]
            }
            with open(self.db_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[auth] Warning: save failed: {e}")

    def create_user(self, username: str, role: UserRole = UserRole.BASIC) -> User:
        user_id = f"user_{secrets.token_hex(8)}"
        api_key = f"sk_{secrets.token_hex(32)}"
        
        user = User(
            user_id=user_id,
            username=username,
            role=role,
            api_key=api_key,
            created_at=time.time(),
        )
        
        self._users[user_id] = user
        self._api_keys[api_key] = user_id
        self._save()
        
        return user

    def get_user_by_api_key(self, api_key: str) -> Optional[User]:
        user_id = self._api_keys.get(api_key)
        if user_id:
            user = self._users.get(user_id)
            if user and user.is_active:
                user.last_login = time.time()
                return user
        return None

    def get_user_by_id(self, user_id: str) -> Optional[User]:
        return self._users.get(user_id)

    def create_session_token(self, user_id: str) -> str:
        token = secrets.token_hex(32)
        self._session_tokens[token] = {
            "user_id": user_id,
            "created_at": time.time(),
            "expires_at": time.time() + 86400,
        }
        return token

    def validate_session_token(self, token: str) -> Optional[User]:
        session = self._session_tokens.get(token)
        if session:
            if time.time() < session["expires_at"]:
                return self.get_user_by_id(session["user_id"])
            else:
                del self._session_tokens[token]
        return None

    def revoke_session(self, token: str):
        self._session_tokens.pop(token, None)

    def update_role(self, user_id: str, new_role: UserRole) -> bool:
        user = self._users.get(user_id)
        if user:
            user.role = new_role
            self._save()
            return True
        return False

    def deactivate_user(self, user_id: str) -> bool:
        user = self._users.get(user_id)
        if user:
            user.is_active = False
            self._save()
            return True
        return False

    def list_users(self) -> list[dict]:
        return [user.to_dict() for user in self._users.values()]

    def get_user_count(self) -> int:
        return len(self._users)

    def create_default_admin(self) -> Optional[User]:
        admin_exists = any(
            u.role == UserRole.ADMIN for u in self._users.values()
        )
        if not admin_exists:
            return self.create_user("admin", UserRole.ADMIN)
        return None


_auth_manager: Optional[AuthManager] = None


def get_auth_manager() -> AuthManager:
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthManager()
        _auth_manager.load()
    return _auth_manager
