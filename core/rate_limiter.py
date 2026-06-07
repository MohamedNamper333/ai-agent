"""Rate Limiter - Token bucket algorithm for API protection"""
import time
import threading
from collections import defaultdict
from typing import Optional


class RateLimiter:
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()
        self._cleanup_interval = 60
        self._last_cleanup = time.time()

    def _cleanup_old_entries(self):
        current_time = time.time()
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
        
        with self._lock:
            cutoff = current_time - self.window_seconds
            for key in list(self._requests.keys()):
                self._requests[key] = [t for t in self._requests[key] if t > cutoff]
                if not self._requests[key]:
                    del self._requests[key]
            self._last_cleanup = current_time

    def is_allowed(self, key: str) -> bool:
        self._cleanup_old_entries()
        
        current_time = time.time()
        cutoff = current_time - self.window_seconds
        
        with self._lock:
            self._requests[key] = [t for t in self._requests[key] if t > cutoff]
            
            if len(self._requests[key]) >= self.max_requests:
                return False
            
            self._requests[key].append(current_time)
            return True

    def get_remaining(self, key: str) -> int:
        current_time = time.time()
        cutoff = current_time - self.window_seconds
        
        with self._lock:
            recent = [t for t in self._requests.get(key, []) if t > cutoff]
            return max(0, self.max_requests - len(recent))

    def get_reset_time(self, key: str) -> Optional[float]:
        with self._lock:
            if not self._requests.get(key):
                return None
            oldest = min(self._requests[key])
            return oldest + self.window_seconds - time.time()


class TieredRateLimiter:
    def __init__(self):
        self._limiters = {
            "anonymous": RateLimiter(max_requests=20, window_seconds=60),
            "basic": RateLimiter(max_requests=60, window_seconds=60),
            "premium": RateLimiter(max_requests=200, window_seconds=60),
            "admin": RateLimiter(max_requests=1000, window_seconds=60),
        }

    def is_allowed(self, key: str, tier: str = "basic") -> bool:
        limiter = self._limiters.get(tier, self._limiters["basic"])
        return limiter.is_allowed(key)

    def get_remaining(self, key: str, tier: str = "basic") -> int:
        limiter = self._limiters.get(tier, self._limiters["basic"])
        return limiter.get_remaining(key)


_rate_limiter: Optional[TieredRateLimiter] = None


def get_rate_limiter() -> TieredRateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = TieredRateLimiter()
    return _rate_limiter
