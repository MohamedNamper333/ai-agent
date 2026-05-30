import subprocess
import os
from pathlib import Path


class GitOps:
    @staticmethod
    def _run_git(args: list[str], cwd: str = "") -> str:
        try:
            cw = cwd or os.getcwd()
            result = subprocess.run(
                ["git"] + args,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=cw,
            )
            output = result.stdout or ""
            if result.stderr:
                stderr = result.stderr.strip()
                if not result.stdout and stderr:
                    return f"Git error: {stderr[:500]}"
                if stderr:
                    output += f"\n(stderr: {stderr[:200]})"
            return output.strip() or "(no output)"
        except FileNotFoundError:
            return "Error: git not found. Install git from https://git-scm.com"
        except subprocess.TimeoutExpired:
            return "Error: git command timed out (30s)"
        except Exception as e:
            return f"Error: {e}"

    @staticmethod
    def git_status(path: str = "") -> str:
        return GitOps._run_git(["status", "--short"], path)

    @staticmethod
    def git_diff(path: str = "", staged: bool = False) -> str:
        args = ["diff"]
        if staged:
            args.append("--staged")
        result = GitOps._run_git(args, path)
        if len(result) > 5000:
            result = result[:5000] + "\n... (diff truncated to 5000 chars)"
        return result

    @staticmethod
    def git_log(path: str = "", count: int = 20) -> str:
        return GitOps._run_git(
            ["log", f"-{count}", "--oneline", "--graph", "--decorate"],
            path,
        )

    @staticmethod
    def git_branch(path: str = "") -> str:
        return GitOps._run_git(["branch", "-a"], path)

    @staticmethod
    def git_show(commit: str = "HEAD", path: str = "") -> str:
        result = GitOps._run_git(["show", commit, "--stat", "--no-patch"], path)
        if len(result) < 2000:
            result = GitOps._run_git(["show", commit], path)
        if len(result) > 5000:
            result = result[:5000] + "\n... (truncated)"
        return result

    @staticmethod
    def is_git_repo(path: str = "") -> str:
        p = Path(path) if path else Path.cwd()
        git_dir = p / ".git"
        if git_dir.exists() and git_dir.is_dir():
            root = GitOps._run_git(["rev-parse", "--show-toplevel"], str(p))
            return f"Git repository: {root}"
        return "Not a git repository"
