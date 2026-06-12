"""gunicorn.conf.py — Production server configuration.

Usage:
    gunicorn web:app --config gunicorn.conf.py

Why gunicorn + uvicorn workers:
    - gunicorn = process manager (restarts crashed workers)
    - uvicorn workers = async ASGI (handles SSE streaming)
    - Together: production-ready, stable, restartable
"""
import multiprocessing
import os

# ── Binding ───────────────────────────────────────────────────
bind = f"0.0.0.0:{os.getenv('WEB_PORT', '8080')}"

# ── Workers ──────────────────────────────────────────────────
# uvicorn worker class for ASGI / async FastAPI
worker_class = "uvicorn.workers.UvicornWorker"

# Formula: (2 × CPU cores) + 1  — standard for I/O-bound apps
workers = int(os.getenv("GUNICORN_WORKERS", multiprocessing.cpu_count() * 2 + 1))

# Threads per worker (for CPU-bound tasks within a worker)
threads = int(os.getenv("GUNICORN_THREADS", "1"))

# ── Timeouts ─────────────────────────────────────────────────
# LLM calls can be slow — set generous timeout
timeout = int(os.getenv("GUNICORN_TIMEOUT", "300"))   # 5 minutes
keepalive = 5
graceful_timeout = 30

# ── Logging ──────────────────────────────────────────────────
loglevel = os.getenv("LOG_LEVEL", "info").lower()
accesslog = "-"     # stdout
errorlog = "-"      # stderr
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s %(D)sµs'

# ── Process ──────────────────────────────────────────────────
preload_app = False   # Set True in production after testing (faster startup)
daemon = False        # Docker manages process lifecycle

# ── SSL (uncomment in Phase 4) ───────────────────────────────
# keyfile = "/etc/nginx/ssl/privkey.pem"
# certfile = "/etc/nginx/ssl/fullchain.pem"

# ── Hooks ────────────────────────────────────────────────────
def on_starting(server):
    print(f"[gunicorn] Starting AI Agent — {workers} workers on {bind}")

def worker_exit(server, worker):
    print(f"[gunicorn] Worker {worker.pid} exited")
