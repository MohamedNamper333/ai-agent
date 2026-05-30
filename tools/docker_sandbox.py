"""Docker sandbox for secure code execution"""

import subprocess
import sys
import tempfile
from pathlib import Path


class DockerSandbox:
    @staticmethod
    def is_available() -> bool:
        try:
            r = subprocess.run(
                ["docker", "--version"],
                capture_output=True, text=True, timeout=5,
            )
            return r.returncode == 0
        except Exception:
            return False

    @staticmethod
    def run_code(code: str, language: str = "python", timeout: int = 30) -> str:
        if not DockerSandbox.is_available():
            return "Error: Docker not available. Use run_code tool instead."

        lang_configs = {
            "python": {
                "image": "python:3.11-slim",
                "cmd": ["python", "-c", code],
            },
            "bash": {
                "image": "ubuntu:22.04",
                "cmd": ["bash", "-c", code],
            },
            "node": {
                "image": "node:20-slim",
                "cmd": ["node", "-e", code],
            },
        }

        cfg = lang_configs.get(language, lang_configs["python"])

        try:
            result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "--network", "none",
                    "--memory", "512m",
                    "--cpus", "1",
                    "--timeout", str(timeout),
                    cfg["image"],
                ] + cfg["cmd"],
                capture_output=True, text=True, timeout=timeout + 10,
            )
            output = result.stdout or ""
            if result.stderr:
                output += f"\nstderr: {result.stderr[:500]}"
            if result.returncode != 0:
                output += f"\nExit code: {result.returncode}"
            return output.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: Execution timed out"
        except FileNotFoundError:
            return "Error: Docker not found"
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def run_file(file_path: str, language: str = "python") -> str:
        p = Path(file_path)
        if not p.exists():
            return f"Error: File not found: {file_path}"
        code = p.read_text(encoding="utf-8", errors="replace")
        return DockerSandbox.run_code(code, language)

    @staticmethod
    def list_images() -> str:
        if not DockerSandbox.is_available():
            return "Docker not available"
        try:
            r = subprocess.run(
                ["docker", "images", "--format", "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"],
                capture_output=True, text=True, timeout=10,
            )
            return r.stdout or "No images found"
        except Exception as e:
            return f"Error: {e}"
