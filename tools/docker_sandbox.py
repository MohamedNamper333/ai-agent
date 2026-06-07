"""Docker Sandbox - isolated code execution"""

import subprocess
import sys
from pathlib import Path
from typing import Optional


class DockerSandbox:
    _available = None

    @classmethod
    def is_available(cls) -> bool:
        if cls._available is not None:
            return cls._available
        try:
            r = subprocess.run(
                ["docker", "--version"],
                capture_output=True, text=True, timeout=5,
            )
            cls._available = r.returncode == 0
        except Exception:
            cls._available = False
        return cls._available

    @classmethod
    def run_code(cls, code: str, language: str = "python", timeout: int = 30) -> str:
        if not cls.is_available():
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
                    "--read-only",
                    "--tmpfs", "/tmp:size=100m",
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

    @classmethod
    def run_file(cls, file_path: str, language: str = "python") -> str:
        p = Path(file_path)
        if not p.exists():
            return f"Error: File not found: {file_path}"
        code = p.read_text(encoding="utf-8", errors="replace")
        return cls.run_code(code, language)

    @classmethod
    def run_with_volume(cls, code: str, volume_path: str, language: str = "python") -> str:
        if not cls.is_available():
            return "Error: Docker not available"

        p = Path(volume_path)
        if not p.exists():
            return f"Error: Volume path not found: {volume_path}"

        lang_configs = {
            "python": {
                "image": "python:3.11-slim",
                "cmd": ["python", "-c", code],
            },
            "bash": {
                "image": "ubuntu:22.04",
                "cmd": ["bash", "-c", code],
            },
        }

        cfg = lang_configs.get(language, lang_configs["python"])

        try:
            result = subprocess.run(
                [
                    "docker", "run", "--rm",
                    "-v", f"{p.absolute()}:/workspace",
                    "-w", "/workspace",
                    "--network", "none",
                    "--memory", "512m",
                    "--cpus", "1",
                    cfg["image"],
                ] + cfg["cmd"],
                capture_output=True, text=True, timeout=60,
            )
            output = result.stdout or ""
            if result.stderr:
                output += f"\nstderr: {result.stderr[:500]}"
            if result.returncode != 0:
                output += f"\nExit code: {result.returncode}"
            return output.strip() or "(no output)"
        except Exception as e:
            return f"Error: {e}"

    @classmethod
    def list_images(cls) -> str:
        if not cls.is_available():
            return "Docker not available"
        try:
            r = subprocess.run(
                ["docker", "images", "--format", "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"],
                capture_output=True, text=True, timeout=10,
            )
            return r.stdout or "No images found"
        except Exception as e:
            return f"Error: {e}"

    @classmethod
    def list_containers(cls, all_containers: bool = True) -> str:
        if not cls.is_available():
            return "Docker not available"
        try:
            args = ["docker", "ps"]
            if all_containers:
                args.append("-a")
            args.extend(["--format", "table {{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Image}}"])

            r = subprocess.run(args, capture_output=True, text=True, timeout=10)
            return r.stdout or "No containers found"
        except Exception as e:
            return f"Error: {e}"

    @classmethod
    def docker_info(cls) -> str:
        if not cls.is_available():
            return "Docker not available"
        try:
            r = subprocess.run(
                ["docker", "info", "--format", "{{.ServerVersion}}"],
                capture_output=True, text=True, timeout=10,
            )
            version = r.stdout.strip() if r.returncode == 0 else "unknown"

            r2 = subprocess.run(
                ["docker", "system", "df"],
                capture_output=True, text=True, timeout=10,
            )

            info = f"Docker version: {version}\n\nDisk usage:\n{r2.stdout}"
            return info
        except Exception as e:
            return f"Error: {e}"

    @classmethod
    def cleanup(cls) -> str:
        if not cls.is_available():
            return "Docker not available"
        try:
            r = subprocess.run(
                ["docker", "system", "prune", "-f"],
                capture_output=True, text=True, timeout=30,
            )
            return r.stdout or "Cleanup completed"
        except Exception as e:
            return f"Error: {e}"
