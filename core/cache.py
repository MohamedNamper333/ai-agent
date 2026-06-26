"""Cache System - In-memory caching for performance"""
import time
import threading
from typing import Any, Optional, Callable
from dataclasses import dataclass, field
from collections import OrderedDict
import hashlib
import json


@dataclass
class CacheEntry:
    key: str
    value: Any
    created_at: float = 0.0
    expires_at: float = 0.0
    access_count: int = 0
    last_accessed: float = 0.0

    def __post_init__(self):
        if not self.created_at:
            self.created_at = time.time()
        if not self.last_accessed:
            self.last_accessed = time.time()

    def is_expired(self) -> bool:
        """Return True if this cache entry has passed its expiry time."""
        if self.expires_at <= 0:
            return False
        return time.time() > self.expires_at

    def touch(self):
        """Update access count and last-accessed timestamp on this entry."""
        self.access_count += 1
        self.last_accessed = time.time()


class LRUCache:
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
        }

    def get(self, key: str) -> Optional[Any]:
        """Return the Tool with the given name, loading its category if needed."""
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._stats["misses"] += 1
                return None
            
            if entry.is_expired():
                del self._cache[key]
                self._stats["misses"] += 1
                return None
            
            entry.touch()
            self._cache.move_to_end(key)
            self._stats["hits"] += 1
            return entry.value

    def set(self, key: str, value: Any, ttl: int = None):
        """Set."""
        ttl = ttl or self.default_ttl
        
        with self._lock:
            if key in self._cache:
                del self._cache[key]
            elif len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)
                self._stats["evictions"] += 1
            
            entry = CacheEntry(
                key=key,
                value=value,
                expires_at=time.time() + ttl if ttl > 0 else 0,
            )
            self._cache[key] = entry

    def delete(self, key: str) -> bool:
        """Remove the entry with the given key. Return True if it existed."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self):
        """Remove all entries from the cache."""
        with self._lock:
            self._cache.clear()

    def cleanup_expired(self):
        """Evict all entries that have passed their expiry time."""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired()
            ]
            for key in expired_keys:
                del self._cache[key]

    def get_or_set(
        self, 
        key: str, 
        factory: Callable, 
        ttl: int = None
    ) -> Any:
        """Return the cached value or call factory, cache it, and return its result."""
        value = self.get(key)
        if value is not None:
            return value
        
        value = factory()
        self.set(key, value, ttl)
        return value

    def get_stats(self) -> dict:
        """Return hit rate, miss count, eviction count, and current size."""
        with self._lock:
            total_requests = self._stats["hits"] + self._stats["misses"]
            hit_rate = (
                self._stats["hits"] / max(total_requests, 1) * 100
            )
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "hit_rate": round(hit_rate, 2),
                "evictions": self._stats["evictions"],
            }

    def keys(self) -> list:
        """Return a list of all keys currently in the cache."""
        with self._lock:
            return list(self._cache.keys())

    def values(self) -> list:
        """Return a list of all non-expired values currently in the cache."""
        with self._lock:
            return [
                entry.value 
                for entry in self._cache.values()
                if not entry.is_expired()
            ]


class CacheManager:
    def __init__(self):
        self._caches: dict[str, LRUCache] = {}

    def get_cache(self, name: str, max_size: int = 1000, ttl: int = 300) -> LRUCache:
        """Return a named LRUCache, creating it with given params if it does not exist."""
        if name not in self._caches:
            self._caches[name] = LRUCache(max_size=max_size, default_ttl=ttl)
        return self._caches[name]

    def clear_all(self):
        """Clear all caches managed by this CacheManager."""
        for cache in self._caches.values():
            cache.clear()

    def cleanup_all(self):
        """Evict expired entries from every cache in the manager."""
        for cache in self._caches.values():
            cache.cleanup_expired()

    def get_stats(self) -> dict:
        """Return hit rate, miss count, eviction count, and current size."""
        return {
            name: cache.get_stats()
            for name, cache in self._caches.items()
        }


def make_cache_key(*args, **kwargs) -> str:
    """Generate a deterministic MD5 cache key from the given arguments."""
    key_parts = [str(arg) for arg in args]
    key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
    key_string = ":".join(key_parts)
    return hashlib.md5(key_string.encode()).hexdigest()


_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Return the global singleton CacheManager, creating it if needed."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def cached(ttl: int = 300, cache_name: str = "default"):
    """Cached."""
    def decorator(func):
        """Decorator."""
        def wrapper(*args, **kwargs):
            """Wrapper."""
            cache = get_cache_manager().get_cache(cache_name, ttl=ttl)
            key = make_cache_key(func.__name__, *args, **kwargs)
            
            result = cache.get(key)
            if result is not None:
                return result
            
            result = func(*args, **kwargs)
            cache.set(key, result, ttl)
            return result
        
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper
    
    return decorator
