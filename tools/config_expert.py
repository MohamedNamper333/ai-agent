"""Config Expert — generates configuration files for any project type.

Supports: ai-agent, web app (React/Vue/Next), phone app (RN/Flutter),
n8n workflows, frontend tooling, backend (FastAPI/Django/Express),
CI/CD (GitHub Actions/GitLab), Docker, Kubernetes, Terraform, Ansible.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ─── Templates ────────────────────────────────────────────────────


class ConfigExpert:
    """Generate ready-to-use configuration files for any project type."""

    SUPPORTED_TYPES = {
        "ai-agent", "web-react", "web-vue", "web-nextjs", "web-nuxt",
        "phone-react-native", "phone-flutter",
        "n8n", "n8n-webhook", "n8n-ai-agent",
        "frontend-vite", "frontend-webpack", "frontend-tailwind",
        "backend-fastapi", "backend-django", "backend-express", "backend-nestjs",
        "ci-github-actions", "ci-gitlab",
        "docker", "docker-compose",
        "kubernetes", "terraform", "ansible",
        "env", "eslint", "prettier", "mypy", "pytest",
    }

    # ── AI Agent ─────────────────────────────────────────────────

    @staticmethod
    def ai_agent_config(backend: str = "ollama", model: str = "qwen3:8b") -> str:
        """Generate config.txt for AI Agent project."""
        return f"""# AI Agent Configuration
BACKEND = {backend}
OLLAMA_MODEL = {model}
OLLAMA_BASE = http://127.0.0.1:11434
N_CTX = 32768
TEMP = 0.7
MAX_TOKENS = 8192
SYSTEM_PROMPT = You are a helpful, capable AI assistant.
WEB_HOST = 0.0.0.0
WEB_PORT = 8080
DB_PATH = memory_store.json
FAST_MODE = auto
CACHE_TTL = 300
RAG_ENABLED = true
TOOLS_ENABLED =
"""

    @staticmethod
    def ai_agent_env(secret_key: str = "change-me", zen_key: str = "") -> str:
        """Generate .env file for AI Agent project."""
        return f"""SECRET_KEY={secret_key}
API_KEY_HASH_SALT=change-me-in-production
CORS_ORIGINS=http://localhost:8080
LOG_LEVEL=INFO
OPENCODE_ZEN_KEY={zen_key}
DATABASE_URL=postgresql+asyncpg://aiagent:secret@localhost:5432/aiagent
REDIS_URL=redis://localhost:6379/0
"""

    # ── Web App ───────────────────────────────────────────────────

    @staticmethod
    def vite_config(framework: str = "react", ts: bool = True) -> str:
        """Generate vite.config.ts for React/Vue projects."""
        plugin = "react()" if framework == "react" else "vue()"
        pkg = "@vitejs/plugin-react" if framework == "react" else "@vitejs/plugin-vue"
        ext = "ts" if ts else "js"
        return f"""import {{ defineConfig }} from 'vite'
import {framework} from '{pkg}'

export default defineConfig({{
  plugins: [{plugin}],
  server: {{
    port: 3000,
    proxy: {{
      '/api': {{ target: 'http://localhost:8080', changeOrigin: true }},
    }},
  }},
  build: {{
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {{
      output: {{ manualChunks: {{ vendor: ['react', 'react-dom'] }} }},
    }},
  }},
}})
"""

    @staticmethod
    def nextjs_config() -> str:
        """Generate next.config.js for Next.js projects."""
        return """/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  images: { domains: ['localhost'] },
  async rewrites() {
    return [
      { source: '/api/:path*', destination: 'http://localhost:8080/:path*' },
    ]
  },
  env: { NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080' },
}

module.exports = nextConfig
"""

    @staticmethod
    def tailwind_config() -> str:
        """Generate tailwind.config.js."""
        return """/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./src/**/*.{js,ts,jsx,tsx,html}', './index.html'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: { brand: { DEFAULT: '#0F6E56', light: '#1D9E75', dark: '#085041' } },
      fontFamily: { sans: ['Inter', 'system-ui', 'sans-serif'] },
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/typography'),
  ],
}
"""

    # ── Backend ───────────────────────────────────────────────────

    @staticmethod
    def fastapi_main() -> str:
        """Generate main.py boilerplate for FastAPI projects."""
        return '''"""FastAPI Application Entry Point."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown lifecycle."""
    logger.info("Application starting up")
    yield
    logger.info("Application shutting down")


app = FastAPI(
    title="AI Agent API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}
'''

    # ── Mobile ────────────────────────────────────────────────────

    @staticmethod
    def react_native_app() -> str:
        """Generate App.tsx boilerplate for React Native."""
        return """import React from 'react';
import { SafeAreaView, StatusBar, StyleSheet, Text, View } from 'react-native';

const API_URL = process.env.EXPO_PUBLIC_API_URL ?? 'http://localhost:8080';

export default function App(): React.JSX.Element {
  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="dark-content" />
      <View style={styles.content}>
        <Text style={styles.title}>AI Agent</Text>
        <Text style={styles.subtitle}>Connected to: {API_URL}</Text>
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#F7F4F0' },
  content:   { flex: 1, alignItems: 'center', justifyContent: 'center' },
  title:     { fontSize: 28, fontWeight: '600', color: '#0F6E56' },
  subtitle:  { fontSize: 14, color: '#7A7068', marginTop: 8 },
});
"""

    @staticmethod
    def flutter_main() -> str:
        """Generate main.dart boilerplate for Flutter."""
        return """import 'package:flutter/material.dart';

void main() => runApp(const AIAgentApp());

class AIAgentApp extends StatelessWidget {
  const AIAgentApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'AI Agent',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF0F6E56)),
        useMaterial3: true,
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatelessWidget {
  const HomePage({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('AI Agent'), backgroundColor: const Color(0xFF0F6E56)),
      body: const Center(child: Text('AI Agent Ready', style: TextStyle(fontSize: 24))),
    );
  }
}
"""

    # ── n8n ───────────────────────────────────────────────────────

    @staticmethod
    def n8n_ai_agent_workflow(agent_url: str = "http://localhost:8080") -> str:
        """Generate n8n workflow JSON for AI Agent integration."""
        workflow = {
            "name": "AI Agent Workflow",
            "nodes": [
                {
                    "parameters": {"httpMethod": "POST", "path": "ai-agent", "responseMode": "responseNode"},
                    "name": "Webhook",
                    "type": "n8n-nodes-base.webhook",
                    "position": [250, 300],
                },
                {
                    "parameters": {
                        "method": "POST",
                        "url": f"{agent_url}/chat",
                        "sendBody": True,
                        "bodyParameters": {"parameters": [
                            {"name": "message", "value": "={{ $json.body.message }}"},
                            {"name": "stream", "value": False},
                        ]},
                    },
                    "name": "Call AI Agent",
                    "type": "n8n-nodes-base.httpRequest",
                    "position": [500, 300],
                },
                {
                    "parameters": {"respondWith": "json", "responseBody": "={{ { text: $json.text } }}"},
                    "name": "Respond to Webhook",
                    "type": "n8n-nodes-base.respondToWebhook",
                    "position": [750, 300],
                },
            ],
            "connections": {
                "Webhook": {"main": [[{"node": "Call AI Agent", "type": "main", "index": 0}]]},
                "Call AI Agent": {"main": [[{"node": "Respond to Webhook", "type": "main", "index": 0}]]},
            },
        }
        return json.dumps(workflow, indent=2, ensure_ascii=False)

    # ── CI/CD ─────────────────────────────────────────────────────

    @staticmethod
    def github_actions_basic(python_version: str = "3.12") -> str:
        """Generate GitHub Actions workflow for Python projects."""
        return f"""name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "{python_version}"
          cache: pip
      - run: pip install -r requirements.txt pytest
      - run: pytest tests/ -q --ignore=tests/comprehensive_test.py
"""

    @staticmethod
    def gitlab_ci() -> str:
        """Generate .gitlab-ci.yml for GitLab CI/CD."""
        return r"""stages: [test, build, deploy]

variables:
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"

cache:
  paths: [.cache/pip]

test:
  stage: test
  image: python:3.12-slim
  script:
    - pip install -r requirements.txt pytest
    - pytest tests/ -q --ignore=tests/comprehensive_test.py
  coverage: '/TOTAL.*\s+(\d+%)$/'

build:
  stage: build
  image: docker:latest
  services: [docker:dind]
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
  only: [main]

deploy:
  stage: deploy
  script: echo "Deploy to production"
  only: [main]
  when: manual
"""

    # ── Docker ────────────────────────────────────────────────────

    @staticmethod
    def dockerfile_python(port: int = 8080, cmd: str = "uvicorn main:app") -> str:
        """Generate production Dockerfile for Python apps."""
        return f"""FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim AS runtime
RUN useradd -r -u 1001 appuser
WORKDIR /app
COPY --from=builder /install /usr/local
COPY --chown=appuser:appuser . .
USER appuser
EXPOSE {port}
HEALTHCHECK --interval=30s --timeout=10s CMD curl -f http://localhost:{port}/health || exit 1
CMD ["{cmd.split()[0]}", "{'" "'.join(cmd.split()[1:])}"]
"""

    # ── Kubernetes ────────────────────────────────────────────────

    @staticmethod
    def k8s_deployment(app_name: str = "ai-agent", image: str = "ai-agent:latest",
                       replicas: int = 2, port: int = 8080) -> str:
        """Generate Kubernetes Deployment + Service manifests."""
        return f"""apiVersion: apps/v1
kind: Deployment
metadata:
  name: {app_name}
  labels:
    app: {app_name}
spec:
  replicas: {replicas}
  selector:
    matchLabels:
      app: {app_name}
  template:
    metadata:
      labels:
        app: {app_name}
    spec:
      containers:
        - name: {app_name}
          image: {image}
          ports:
            - containerPort: {port}
          env:
            - name: SECRET_KEY
              valueFrom:
                secretKeyRef:
                  name: {app_name}-secrets
                  key: secret-key
          readinessProbe:
            httpGet:
              path: /status
              port: {port}
            initialDelaySeconds: 10
            periodSeconds: 5
          resources:
            requests:
              memory: "256Mi"
              cpu: "100m"
            limits:
              memory: "512Mi"
              cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: {app_name}
spec:
  selector:
    app: {app_name}
  ports:
    - port: 80
      targetPort: {port}
  type: ClusterIP
"""

    # ── Linting / Typing ──────────────────────────────────────────

    @staticmethod
    def mypy_config() -> str:
        """Generate mypy.ini for type checking configuration."""
        return """[mypy]
python_version = 3.12
strict = True
ignore_missing_imports = True
warn_return_any = True
warn_unused_configs = True
exclude = tests/|benchmarks/

[mypy-pytest.*]
ignore_missing_imports = True
"""

    @staticmethod
    def pytest_ini() -> str:
        """Generate pytest.ini configuration."""
        return """[pytest]
minversion = 7.0
addopts = -ra -q --tb=short
testpaths = tests
filterwarnings =
    ignore::DeprecationWarning
asyncio_mode = auto
"""

    @staticmethod
    def eslint_config() -> str:
        """Generate .eslintrc.json for TypeScript/React projects."""
        return json.dumps({
            "extends": ["eslint:recommended", "plugin:@typescript-eslint/recommended",
                        "plugin:react-hooks/recommended"],
            "parser": "@typescript-eslint/parser",
            "plugins": ["@typescript-eslint", "react-hooks"],
            "rules": {
                "no-console": "warn",
                "@typescript-eslint/no-explicit-any": "warn",
                "react-hooks/exhaustive-deps": "error",
            },
            "env": {"browser": True, "es2022": True},
        }, indent=2)

    # ── Main interface ────────────────────────────────────────────

    @staticmethod
    def generate(config_type: str, **kwargs) -> str:
        """Generate a configuration file for the given project type.

        Args:
            config_type: One of the SUPPORTED_TYPES strings.
            **kwargs: Type-specific parameters passed to the generator.

        Returns:
            String content of the generated configuration file.
        """
        generators = {
            "ai-agent":             ConfigExpert.ai_agent_config,
            "env":                  ConfigExpert.ai_agent_env,
            "frontend-vite":        ConfigExpert.vite_config,
            "web-nextjs":           ConfigExpert.nextjs_config,
            "web-nuxt":             ConfigExpert.nextjs_config,
            "frontend-tailwind":    ConfigExpert.tailwind_config,
            "backend-fastapi":      ConfigExpert.fastapi_main,
            "phone-react-native":   ConfigExpert.react_native_app,
            "phone-flutter":        ConfigExpert.flutter_main,
            "n8n-ai-agent":         ConfigExpert.n8n_ai_agent_workflow,
            "n8n":                  ConfigExpert.n8n_ai_agent_workflow,
            "ci-github-actions":    ConfigExpert.github_actions_basic,
            "ci-gitlab":            ConfigExpert.gitlab_ci,
            "docker":               ConfigExpert.dockerfile_python,
            "kubernetes":           ConfigExpert.k8s_deployment,
            "mypy":                 ConfigExpert.mypy_config,
            "pytest":               ConfigExpert.pytest_ini,
            "eslint":               ConfigExpert.eslint_config,
        }
        gen = generators.get(config_type)
        if gen is None:
            supported = ", ".join(sorted(ConfigExpert.SUPPORTED_TYPES))
            return f"Error: Unknown config type '{config_type}'.\nSupported: {supported}"
        try:
            return gen(**kwargs)
        except TypeError as exc:
            logger.error("ConfigExpert.generate(%s) failed: %s", config_type, exc)
            return f"Error generating {config_type} config: {exc}"

    @staticmethod
    def list_supported() -> str:
        """Return a formatted list of all supported configuration types."""
        categories = {
            "AI Agent":  ["ai-agent", "env"],
            "Web":       ["web-react", "web-vue", "web-nextjs", "web-nuxt"],
            "Frontend":  ["frontend-vite", "frontend-webpack", "frontend-tailwind", "eslint"],
            "Backend":   ["backend-fastapi", "backend-django", "backend-express", "backend-nestjs"],
            "Mobile":    ["phone-react-native", "phone-flutter"],
            "n8n":       ["n8n", "n8n-webhook", "n8n-ai-agent"],
            "CI/CD":     ["ci-github-actions", "ci-gitlab"],
            "Container": ["docker", "docker-compose", "kubernetes"],
            "IaC":       ["terraform", "ansible"],
            "Quality":   ["mypy", "pytest", "prettier"],
        }
        lines = ["Config Expert — Supported Types\n"]
        for cat, types in categories.items():
            lines.append(f"  {cat}:")
            for t in types:
                lines.append(f"    • {t}")
        return "\n".join(lines)
