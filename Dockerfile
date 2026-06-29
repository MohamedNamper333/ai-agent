# ── Stage 1: dependencies ──────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build
COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: production image ──────────────────────────────────────
FROM python:3.12-slim AS runtime

# Security: non-root user
RUN groupadd -r aiagent && useradd -r -g aiagent aiagent

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application
COPY --chown=aiagent:aiagent . .

# Remove dev/test files from production image
RUN rm -rf tests/ benchmarks/ .git/ *.md docs/plans/ \
           benchmark_*.py learning_data/ obsidian_vault/

# Create necessary dirs with correct ownership
RUN mkdir -p logs models learning_data obsidian_vault \
 && chown -R aiagent:aiagent /app

USER aiagent

EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8080/status || exit 1

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LOG_LEVEL=INFO

CMD ["python", "-m", "uvicorn", "web:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
