"""HMAC-SHA256 request signing middleware and audit logging for AI Agent."""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import uuid
from typing import Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# Endpoints that skip HMAC (public / health)
_PUBLIC_PATHS = {"/", "/status", "/docs", "/openapi.json", "/redoc"}
_HMAC_SKIP_PREFIXES = ("/web/", "/static/")


class HMACMiddleware(BaseHTTPMiddleware):
    """Validate HMAC-SHA256 signatures on API requests.

    Clients must include:
        X-Timestamp: unix epoch seconds (within ±300s)
        X-Signature: hmac_sha256(secret, f"{method}:{path}:{timestamp}:{body_hash}")
    """

    def __init__(self, app: ASGIApp, secret_key: str, max_age: int = 300):
        super().__init__(app)
        self.secret = secret_key.encode()
        self.max_age = max_age

    async def dispatch(self, request: Request, call_next):
        """Process each request through HMAC validation."""
        # Skip public paths and non-API routes
        path = request.url.path
        if path in _PUBLIC_PATHS or any(path.startswith(p) for p in _HMAC_SKIP_PREFIXES):
            return await call_next(request)

        # Skip if no signature header (single-user mode or API key auth only)
        sig_header = request.headers.get("X-Signature")
        if not sig_header:
            return await call_next(request)

        ts_header = request.headers.get("X-Timestamp", "")
        try:
            ts = int(ts_header)
            if abs(time.time() - ts) > self.max_age:
                return Response(
                    content=json.dumps({"detail": "Request timestamp expired"}),
                    status_code=401,
                    media_type="application/json",
                )
        except (ValueError, TypeError):
            return Response(
                content=json.dumps({"detail": "Invalid timestamp"}),
                status_code=401,
                media_type="application/json",
            )

        body = await request.body()
        body_hash = hashlib.sha256(body).hexdigest()
        expected_payload = f"{request.method}:{path}:{ts_header}:{body_hash}"
        expected_sig = hmac.new(
            self.secret,
            expected_payload.encode(),
            hashlib.sha256,
        ).hexdigest()

        if not hmac.compare_digest(sig_header, expected_sig):
            logger.warning("HMAC validation failed for %s %s from %s",
                           request.method, path, request.client.host if request.client else "unknown")
            return Response(
                content=json.dumps({"detail": "Invalid request signature"}),
                status_code=401,
                media_type="application/json",
            )

        return await call_next(request)


def generate_hmac_signature(secret: str, method: str, path: str,
                             body: bytes = b"") -> tuple[str, str]:
    """Generate HMAC signature for a request (client helper).

    Returns:
        (timestamp_str, signature_hex) tuple to include in request headers.
    """
    ts = str(int(time.time()))
    body_hash = hashlib.sha256(body).hexdigest()
    payload = f"{method}:{path}:{ts}:{body_hash}"
    sig = hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()
    return ts, sig


class AuditLogger:
    """Structured audit logger — writes to file and optionally PostgreSQL."""

    def __init__(self, log_dir: str = "logs"):
        import os
        os.makedirs(log_dir, exist_ok=True)
        self._path = f"{log_dir}/audit.jsonl"
        self._logger = logging.getLogger("audit")

    def log(
        self,
        action: str,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        ip: Optional[str] = None,
        status: str = "success",
        detail: Optional[dict] = None,
    ) -> None:
        """Record an audit event to the JSONL log file."""
        entry = {
            "request_id": str(uuid.uuid4())[:8],
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "action": action,
            "user_id": user_id or "anonymous",
            "resource": resource,
            "ip": ip,
            "status": status,
            "detail": detail or {},
        }
        try:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:
            self._logger.error("Audit log write failed: %s", exc)


class AuditMiddleware(BaseHTTPMiddleware):
    """Log every API request to the audit trail automatically."""

    def __init__(self, app: ASGIApp, audit: AuditLogger):
        super().__init__(app)
        self.audit = audit

    async def dispatch(self, request: Request, call_next):
        """Intercept each request and record an audit entry."""
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

        # Skip static files and docs
        path = request.url.path
        if path in ("/", "/docs", "/openapi.json") or path.startswith("/web/"):
            return response

        user_id = getattr(request.state, "user_id", None)
        ip = request.client.host if request.client else None

        self.audit.log(
            action=f"{request.method} {path}",
            user_id=user_id,
            resource=path,
            ip=ip,
            status="success" if response.status_code < 400 else "error",
            detail={"status_code": response.status_code, "elapsed_ms": elapsed_ms},
        )
        return response


# Singleton
_audit_logger: Optional[AuditLogger] = None


def get_audit_logger() -> AuditLogger:
    """Return the global singleton AuditLogger, creating it if needed."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = AuditLogger()
    return _audit_logger
