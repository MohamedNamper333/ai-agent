"""Benchmark: _DynamicCORSMiddleware per-request CORSMiddleware rebuild cost.

Drives a configurable number of HTTP preflight requests through an ASGI
harness that wraps an inner app with the supplied middleware class. The
inner app is trivial (just sends 200 OK with no body), so the time
captured is dominated by middleware overhead.

Usage:
    python benchmark_cors_middleware.py [classname] [iterations]

Defaults: 5000 iterations against whichever class is currently defined
in web.py as ``_DynamicCORSMiddleware``.
"""

import sys
import os
import time
import asyncio
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.cors import CORSMiddleware
from web import _resolve_cors_config, _DynamicCORSMiddleware  # noqa: E402


INNER_APP = FastAPI()


@INNER_APP.get("/ping")
def _ping():  # noqa: D401
    return "ok"


def _build():
    """Compose a FastAPI app whose CORS layer is the *supplied* class."""
    a = FastAPI()

    @a.get("/ping")
    def _ping():  # noqa: F811
        return "ok"

    a.add_middleware(_DynamicCORSMiddleware)
    return a


def _run(n: int) -> float:
    """Drive ``n`` preflight requests; return total wall time (seconds)."""
    app = _build()
    client = TestClient(app, raise_server_exceptions=False)
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
    p.add_argument("classname", nargs="?", default="_DynamicCORSMiddleware")
    p.add_argument("iterations", nargs="?", type=int, default=5000)
    args = p.parse_args()

    # Resolve class object from web module
    import web
    cls = getattr(web, args.classname)
    globals()["_DynamicCORSMiddleware"] = cls  # used by _build via name

    # Warm up (one request) so first-call imports/disk-cache settle.
    _run(50)

    elapsed = _run(args.iterations)
    per_req_us = (elapsed / args.iterations) * 1e6
    print(
        f"class={args.classname}  iters={args.iterations}  "
        f"total={elapsed:.4f}s  per_req={per_req_us:.2f}us"
    )


if __name__ == "__main__":
    main()
