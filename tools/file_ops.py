import os
import re
from pathlib import Path


class FileOps:
    @staticmethod
    def read_file(path: str, offset: int = 0, limit: int = 2000) -> str:
        p = Path(path)
        if not p.exists():
            return f"Error: File not found: {path}"
        if not p.is_file():
            return f"Error: Not a file: {path}"
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
            total = len(lines)
            start = offset
            end = min(offset + limit, total)
            selected = lines[start:end]
            result = "".join(selected)
            info = f"File: {path} ({total} lines)"
            if limit < total:
                info += f" [showing {start+1}-{end} of {total}]"
            return f"{info}\n{'-'*40}\n{result}"
        except Exception as e:
            return f"Error reading file: {e}"

    @staticmethod
    def write_file(path: str, content: str) -> str:
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"Written {len(content)} bytes to {path}"
        except Exception as e:
            return f"Error writing file: {e}"

    @staticmethod
    def edit_file(path: str, old_string: str, new_string: str) -> str:
        p = Path(path)
        if not p.exists():
            return f"Error: File not found: {path}"
        try:
            content = p.read_text(encoding="utf-8")
            if old_string not in content:
                return f"Error: old_string not found in file"
            count = content.count(old_string)
            content = content.replace(old_string, new_string)
            p.write_text(content, encoding="utf-8")
            return f"Replaced {count} occurrence(s) in {path}"
        except Exception as e:
            return f"Error editing file: {e}"

    @staticmethod
    def glob_search(pattern: str, path: str = "") -> str:
        search_path = Path(path) if path else Path.cwd()
        if not search_path.exists():
            return f"Error: Path not found: {path}"
        try:
            matches = [str(p.relative_to(search_path)) for p in search_path.rglob(pattern)]
            if not matches:
                return f"No files matching '{pattern}' in {search_path}"
            return f"Matches for '{pattern}' ({len(matches)}):\n" + "\n".join(matches[:100])
        except Exception as e:
            return f"Error searching: {e}"

    @staticmethod
    def grep_search(pattern: str, path: str = "", include: str = "*") -> str:
        search_path = Path(path) if path else Path.cwd()
        if not search_path.exists():
            return f"Error: Path not found: {path}"
        try:
            results = []
            for p in search_path.rglob(include):
                if not p.is_file():
                    continue
                try:
                    for i, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                        if re.search(pattern, line, re.IGNORECASE):
                            rel = p.relative_to(search_path)
                            results.append(f"{rel}:{i}: {line.strip()[:120]}")
                except Exception:
                    continue
                if len(results) >= 100:
                    results.append(f"... truncated at 100 results")
                    break
            if not results:
                return f"No matches for '{pattern}' in {search_path}"
            return f"Grep '{pattern}' ({len(results)}):\n" + "\n".join(results)
        except Exception as e:
            return f"Error grepping: {e}"

    @staticmethod
    def list_directory(path: str = "") -> str:
        p = Path(path) if path else Path.cwd()
        if not p.exists():
            return f"Error: Path not found: {path}"
        if not p.is_dir():
            return f"Error: Not a directory: {path}"
        try:
            entries = []
            for entry in sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
                if entry.is_dir():
                    entries.append(f"  {entry.name}/")
                else:
                    size = entry.stat().st_size
                    if size < 1024:
                        size_str = f"{size}B"
                    elif size < 1024 * 1024:
                        size_str = f"{size/1024:.1f}KB"
                    else:
                        size_str = f"{size/1024/1024:.1f}MB"
                    entries.append(f"  {entry.name} ({size_str})")
            return f"Directory: {p} ({len(entries)} items)\n" + "\n".join(entries)
        except Exception as e:
            return f"Error listing directory: {e}"

    @staticmethod
    def file_info(path: str) -> str:
        p = Path(path)
        if not p.exists():
            return f"Error: File not found: {path}"
        import datetime
        s = p.stat()
        return (
            f"Path: {p.absolute()}\n"
            f"Size: {s.st_size:,} bytes\n"
            f"Modified: {datetime.datetime.fromtimestamp(s.st_mtime)}\n"
            f"Created: {datetime.datetime.fromtimestamp(s.st_ctime)}\n"
            f"Is dir: {p.is_dir()}"
        )
