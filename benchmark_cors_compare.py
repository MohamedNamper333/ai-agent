"""Side-by-side benchmark: per-request rebuild vs keyed-cache for the CORS wrapper.

Drives N HTTP preflight requests through two ASGI middlewares that wrap the
same trivial inner app, alternating order to amortize warm-up effects:

  • ``_DynamicCORSMiddleware_PerRequest``  — current implementation, builds
    a fresh ``CORSMiddleware`` instance on every HTTP request.
  • ``_DynamicCORSMiddleware_Cached``      — proposed implementation, keys
    on ``(tuple(origins), credentials)`` and reuses the last built
    ``CORSMiddleware`` until the key changes.

The two classes are defined inline so the comparison runs in the same
Python process and the only difference is the cache strategy.

Usage:
    python benchmark_cors_compare.py [iterations]
"""

import sys
import os
import time
import statistics
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.cors import CORSMiddleware
from web import _resolve_cors_config  # noqa: E402


# --- The two implementations ---------------------------------------------

class _DynamicCORSMiddleware_PerRequest:
    """Builds a fresh CORSMiddleware on every HTTP request (current)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        origins, credentials = _resolve_cors_config()
        cors_app = CORSMiddleware(
            self.app,
            allow_origins=origins,
            allow_credentials=credentials,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        await cors_app(scope, receive, send)


class _DynamicCORSMiddleware_Cached:
    """Caches the CORSMiddleware by (origins, credentials) tuple key."""

    def __init__(self, app):
        self.app = app
        self._cache_key = None
        self._cached_cors = None

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        origins, credentials = _resolve_cors_config()
        key = (tuple(origins), credentials)
        if key != self._cache_key:
            self._cached_cors = CORSMiddleware(
                self.app,
                allow_origins=origins,
                allow_credentials=credentials,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            self._cache_key = key
        await self._cached_cors(scope, receive, send)


# --- Benchmark harness ---------------------------------------------------

def _build_app(middleware_cls):
    a = FastAPI()

    @a.get("/ping")
    def _ping():  # noqa: F811
        return "ok"

    a.add_middleware(middleware_cls)
    return a


def _drive(app, n: int) -> float:
    client = TestClient(app, raise_server_exceptions=False)
    # Warm up
    for _ in range(20):
        client.options(
            "/ping",
            headers={
                "Origin": "http://localhost:8080",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
    t0 = time.perf_counter()
    for _ in range(n):
        client.options(
            "/ping",
            headers={
                "Origin": "http://localhost:8080",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
    return time.perf_counter() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("iterations", nargs="?", type=int, default=3000)
    p.add_argument("--repeats", type=int, default=3,
                   help="Number of timed runs per implementation")
    args = p.parse_args()

    app_pr = _build_app(_DynamicCORSMiddleware_PerRequest)
    app_ca = _build_app(_DynamicCORSMiddleware_Cached)

    times_pr = []
    times_ca = []
    for _ in range(args.repeats):
        # Alternate order to amortize drift
        times_ca.append(_drive(app_ca, args.iterations))
        times_pr.append(_drive(app_pr, args.iterations))

    def _report(label, ts):
        med = statistics.median(ts) * 1e6 / args.iterations
        mean = statistics.mean(ts) * 1e6 / args.iterations
        best = min(ts) * 1e6 / args.iterations
        print(f"  {label:>20}  median={med:7.2f}us  mean={mean:7.2f}us  "
              f"best={best:7.2f}us  ({len(ts)} runs of {args.iterations})")

    print(f"CORS middleware benchmark — {args.iterations} preflights per run")
    _report("per_request", times_pr)
    _report("cached", times_ca)

    med_pr = statistics.median(times_pr) * 1e6 / args.iterations
    med_ca = statistics.median(times_ca) * 1e6 / args.iterations
    if med_pr > 0:
        saving = (med_pr - med_ca) / med_pr * 100
        speedup = med_pr / med_ca
        print(f"\n  dMedian per-request latency: -{saving:.1f}%  (speedup x{speedup:.2f})")


if __name__ == "__main__":
    main()
