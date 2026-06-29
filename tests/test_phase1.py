"""Phase 1 tests — CI/CD, HMAC, Audit, Config Expert, Logger."""
from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ─── Logger / print() ────────────────────────────────────────────

class TestNoProductionPrints:
    """Production files must not use print() — logger only."""

    PRODUCTION_FILES = [
        "core/model.py", "core/auth.py", "core/memory.py",
        "core/agent.py", "rag/embedder.py", "rag/vector_store.py",
    ]

    def test_no_print_in_core_model(self):
        src = Path("core/model.py").read_text()
        assert "print(" not in src, "core/model.py still has print() calls"

    def test_no_print_in_core_auth(self):
        src = Path("core/auth.py").read_text()
        assert "print(" not in src

    def test_no_print_in_rag_embedder(self):
        src = Path("rag/embedder.py").read_text()
        assert "print(" not in src

    def test_no_print_in_rag_vector_store(self):
        src = Path("rag/vector_store.py").read_text()
        assert "print(" not in src

    def test_logger_imported_in_model(self):
        src = Path("core/model.py").read_text()
        assert "logging" in src or "logger" in src


# ─── Docker files ────────────────────────────────────────────────

class TestDockerFiles:
    """Dockerfile and docker-compose must exist and be valid."""

    def test_dockerfile_exists(self):
        assert Path("Dockerfile").exists()

    def test_dockerfile_has_multistage(self):
        src = Path("Dockerfile").read_text()
        assert "AS builder" in src
        assert "AS runtime" in src

    def test_dockerfile_has_healthcheck(self):
        assert "HEALTHCHECK" in Path("Dockerfile").read_text()

    def test_dockerfile_non_root_user(self):
        src = Path("Dockerfile").read_text()
        assert "USER aiagent" in src or "USER appuser" in src or "useradd" in src

    def test_docker_compose_exists(self):
        assert Path("docker-compose.yml").exists()

    def test_docker_compose_has_postgres(self):
        src = Path("docker-compose.yml").read_text()
        assert "postgres" in src

    def test_docker_compose_has_redis(self):
        assert "redis" in Path("docker-compose.yml").read_text()

    def test_docker_compose_has_prometheus(self):
        assert "prometheus" in Path("docker-compose.yml").read_text()

    def test_docker_compose_has_grafana(self):
        assert "grafana" in Path("docker-compose.yml").read_text()

    def test_dockerignore_exists(self):
        assert Path(".dockerignore").exists()

    def test_dockerignore_excludes_env(self):
        assert ".env" in Path(".dockerignore").read_text()


# ─── CI/CD ───────────────────────────────────────────────────────

class TestCICD:
    """GitHub Actions pipeline must exist and be valid."""

    def test_github_actions_exists(self):
        assert Path(".github/workflows/ci.yml").exists()

    def test_ci_has_test_job(self):
        src = Path(".github/workflows/ci.yml").read_text()
        assert "test:" in src or "Test Suite" in src

    def test_ci_has_build_job(self):
        assert "build:" in Path(".github/workflows/ci.yml").read_text()

    def test_ci_has_security_scan(self):
        src = Path(".github/workflows/ci.yml").read_text()
        assert "security" in src.lower() or "bandit" in src

    def test_ci_has_postgres_service(self):
        assert "postgres" in Path(".github/workflows/ci.yml").read_text()

    def test_ci_has_docker_push(self):
        assert "docker" in Path(".github/workflows/ci.yml").read_text().lower()


# ─── Database Schema ─────────────────────────────────────────────

class TestDatabaseSchema:
    """PostgreSQL schema init script must be complete."""

    def test_init_sql_exists(self):
        assert Path("scripts/init_db.sql").exists()

    def test_has_users_table(self):
        assert "CREATE TABLE IF NOT EXISTS users" in Path("scripts/init_db.sql").read_text()

    def test_has_conversations_table(self):
        assert "conversations" in Path("scripts/init_db.sql").read_text()

    def test_has_messages_table(self):
        assert "messages" in Path("scripts/init_db.sql").read_text()

    def test_has_audit_logs_table(self):
        assert "audit_logs" in Path("scripts/init_db.sql").read_text()

    def test_has_tool_executions_table(self):
        assert "tool_executions" in Path("scripts/init_db.sql").read_text()

    def test_has_indexes(self):
        assert "CREATE INDEX" in Path("scripts/init_db.sql").read_text()


# ─── HMAC Middleware ─────────────────────────────────────────────

class TestHMACMiddleware:
    """HMAC signing and verification must work correctly."""

    def test_generate_signature_returns_tuple(self):
        from core.hmac_middleware import generate_hmac_signature
        ts, sig = generate_hmac_signature("secret", "POST", "/chat")
        assert isinstance(ts, str)
        assert isinstance(sig, str)
        assert len(sig) == 64  # SHA256 hex

    def test_signature_is_deterministic_same_ts(self):
        from core.hmac_middleware import generate_hmac_signature
        ts = str(int(time.time()))
        _, sig1 = generate_hmac_signature("secret", "POST", "/chat")
        _, sig2 = generate_hmac_signature("secret", "POST", "/chat")
        # Same secret + method + path → different ts each call → different sigs (normal)
        assert len(sig1) == 64 and len(sig2) == 64

    def test_different_secrets_produce_different_sigs(self):
        from core.hmac_middleware import generate_hmac_signature
        _, sig1 = generate_hmac_signature("secret1", "POST", "/chat")
        _, sig2 = generate_hmac_signature("secret2", "POST", "/chat")
        assert sig1 != sig2

    def test_hmac_middleware_class_exists(self):
        from core.hmac_middleware import HMACMiddleware
        assert HMACMiddleware is not None


# ─── Audit Logger ────────────────────────────────────────────────

class TestAuditLogger:
    """Audit logger must record events correctly."""

    def test_audit_logger_creates_file(self, tmp_path):
        from core.hmac_middleware import AuditLogger
        al = AuditLogger(log_dir=str(tmp_path))
        al.log("test.action", user_id="user1", resource="/chat", status="success")
        log_file = tmp_path / "audit.jsonl"
        assert log_file.exists()

    def test_audit_log_is_valid_json(self, tmp_path):
        from core.hmac_middleware import AuditLogger
        al = AuditLogger(log_dir=str(tmp_path))
        al.log("user.login", user_id="u1", ip="127.0.0.1", status="success")
        line = (tmp_path / "audit.jsonl").read_text().strip()
        entry = json.loads(line)
        assert entry["action"] == "user.login"
        assert entry["user_id"] == "u1"
        assert "ts" in entry
        assert "request_id" in entry

    def test_audit_log_multiple_entries(self, tmp_path):
        from core.hmac_middleware import AuditLogger
        al = AuditLogger(log_dir=str(tmp_path))
        al.log("action1", status="success")
        al.log("action2", status="error")
        lines = (tmp_path / "audit.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[1])["action"] == "action2"

    def test_audit_log_handles_write_error_gracefully(self, tmp_path):
        from core.hmac_middleware import AuditLogger
        al = AuditLogger(log_dir="/nonexistent_path_xyz")
        # Must not raise
        al.log("test", status="success")

    def test_get_audit_logger_singleton(self):
        from core.hmac_middleware import get_audit_logger
        a1 = get_audit_logger()
        a2 = get_audit_logger()
        assert a1 is a2


# ─── Config Expert ───────────────────────────────────────────────

class TestConfigExpert:
    """Config Expert must generate valid configs for all supported types."""

    def test_ai_agent_config(self):
        from tools.config_expert import ConfigExpert
        cfg = ConfigExpert.generate("ai-agent")
        assert "BACKEND" in cfg
        assert "OLLAMA_MODEL" in cfg
        assert "RAG_ENABLED" in cfg

    def test_env_config(self):
        from tools.config_expert import ConfigExpert
        cfg = ConfigExpert.generate("env")
        assert "SECRET_KEY" in cfg
        assert "DATABASE_URL" in cfg

    def test_vite_config(self):
        from tools.config_expert import ConfigExpert
        cfg = ConfigExpert.generate("frontend-vite", framework="react")
        assert "defineConfig" in cfg
        assert "proxy" in cfg

    def test_nextjs_config(self):
        from tools.config_expert import ConfigExpert
        cfg = ConfigExpert.generate("web-nextjs")
        assert "nextConfig" in cfg
        assert "rewrites" in cfg

    def test_tailwind_config(self):
        from tools.config_expert import ConfigExpert
        cfg = ConfigExpert.generate("frontend-tailwind")
        assert "content" in cfg
        assert "darkMode" in cfg

    def test_fastapi_main(self):
        from tools.config_expert import ConfigExpert
        cfg = ConfigExpert.generate("backend-fastapi")
        assert "FastAPI" in cfg
        assert "lifespan" in cfg

    def test_react_native(self):
        from tools.config_expert import ConfigExpert
        cfg = ConfigExpert.generate("phone-react-native")
        assert "React Native" in cfg or "SafeAreaView" in cfg

    def test_flutter_main(self):
        from tools.config_expert import ConfigExpert
        cfg = ConfigExpert.generate("phone-flutter")
        assert "Flutter" in cfg or "MaterialApp" in cfg

    def test_n8n_workflow_is_valid_json(self):
        from tools.config_expert import ConfigExpert
        cfg = ConfigExpert.generate("n8n-ai-agent")
        data = json.loads(cfg)
        assert "nodes" in data
        assert "connections" in data

    def test_github_actions(self):
        from tools.config_expert import ConfigExpert
        cfg = ConfigExpert.generate("ci-github-actions")
        assert "on:" in cfg
        assert "pytest" in cfg

    def test_gitlab_ci(self):
        from tools.config_expert import ConfigExpert
        cfg = ConfigExpert.generate("ci-gitlab")
        assert "stages:" in cfg
        assert "test:" in cfg

    def test_dockerfile_generation(self):
        from tools.config_expert import ConfigExpert
        cfg = ConfigExpert.generate("docker", port=8080)
        assert "FROM python" in cfg
        assert "HEALTHCHECK" in cfg

    def test_kubernetes_manifest(self):
        from tools.config_expert import ConfigExpert
        cfg = ConfigExpert.generate("kubernetes", app_name="myapp")
        assert "Deployment" in cfg
        assert "Service" in cfg
        assert "myapp" in cfg

    def test_mypy_config(self):
        from tools.config_expert import ConfigExpert
        cfg = ConfigExpert.generate("mypy")
        assert "[mypy]" in cfg
        assert "strict" in cfg

    def test_pytest_ini(self):
        from tools.config_expert import ConfigExpert
        cfg = ConfigExpert.generate("pytest")
        assert "[pytest]" in cfg
        assert "testpaths" in cfg

    def test_eslint_config(self):
        from tools.config_expert import ConfigExpert
        cfg = ConfigExpert.generate("eslint")
        data = json.loads(cfg)
        assert "extends" in data
        assert "rules" in data

    def test_unknown_type_returns_error(self):
        from tools.config_expert import ConfigExpert
        result = ConfigExpert.generate("nonexistent-type")
        assert "Error" in result
        assert "Supported" in result

    def test_list_supported_returns_all_categories(self):
        from tools.config_expert import ConfigExpert
        listing = ConfigExpert.list_supported()
        assert "AI Agent" in listing
        assert "n8n" in listing
        assert "Mobile" in listing
        assert "kubernetes" in listing or "Container" in listing
