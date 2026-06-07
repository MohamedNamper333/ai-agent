import os
import re
from pathlib import Path
from core.utils import validate_path


class FileOps:
    def __init__(self, base_dir: str = ""):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd().resolve()

    @staticmethod
    def read_file(path: str, offset: int = 0, limit: int = 2000) -> str:
        try:
            p = validate_path(path)
            
            if not p.exists():
                return f"Error: File not found: {path}"
            if not p.is_file():
                return f"Error: Not a file: {path}"
            
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
        except PermissionError as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error reading file: {e}"

    @staticmethod
    def write_file(path: str, content: str) -> str:
        try:
            p = validate_path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"Written {len(content)} bytes to {path}"
        except (PermissionError, FileNotFoundError) as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error writing file: {e}"

    @staticmethod
    def edit_file(path: str, old_string: str, new_string: str) -> str:
        try:
            p = validate_path(path)
            if not p.exists():
                return f"Error: File not found: {path}"
            content = p.read_text(encoding="utf-8")
            if old_string not in content:
                return f"Error: old_string not found in file"
            count = content.count(old_string)
            p.write_text(content.replace(old_string, new_string), encoding="utf-8")
            return f"Replaced {count} occurrence(s) in {path}"
        except (PermissionError, FileNotFoundError) as e:
            return f"Error: {e}"
        except Exception as e:
            return f"Error editing file: {e}"

    @staticmethod
    def glob_search(pattern: str, path: str = "") -> str:
        try:
            search_path = Path(path) if path else Path.cwd()
            if not search_path.exists():
                return f"Error: Path not found: {path}"
            
            matches = [str(p.relative_to(search_path)) for p in search_path.rglob(pattern)]
            if not matches:
                return f"No files matching '{pattern}' in {search_path}"
            return f"Matches for '{pattern}' ({len(matches)}):\n" + "\n".join(matches[:100])
        except Exception as e:
            return f"Error searching: {e}"

    @staticmethod
    def grep_search(pattern: str, path: str = "", include: str = "*") -> str:
        try:
            search_path = Path(path) if path else Path.cwd()
            if not search_path.exists():
                return f"Error: Path not found: {path}"
            
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
        try:
            p = Path(path) if path else Path.cwd()
            if not p.exists():
                return f"Error: Path not found: {path}"
            if not p.is_dir():
                return f"Error: Not a directory: {path}"
            
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
        try:
            p = validate_path(path)
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
        except (PermissionError, FileNotFoundError) as e:
            return f"Error: {e}"

    @staticmethod
    def file_compare(file1: str, file2: str) -> str:
        try:
            p1 = validate_path(file1)
            p2 = validate_path(file2)
            if not p1.exists():
                return f"Error: File not found: {file1}"
            if not p2.exists():
                return f"Error: File not found: {file2}"
            c1 = p1.read_text(encoding="utf-8", errors="replace").splitlines()
            c2 = p2.read_text(encoding="utf-8", errors="replace").splitlines()
            diffs = []
            for i, (l1, l2) in enumerate(zip(c1, c2)):
                if l1 != l2:
                    diffs.append(f"Line {i+1}:\n  - {l1[:100]}\n  + {l2[:100]}")
            if len(c1) != len(c2):
                diffs.append(f"File lengths differ: {len(c1)} vs {len(c2)}")
            if not diffs:
                return f"Files are identical: {file1} and {file2}"
            return f"Comparison: {file1} vs {file2}\nDifferences: {len(diffs)}\n" + "\n".join(diffs)
        except (PermissionError, FileNotFoundError) as e:
            return f"Error: {e}"

    @staticmethod
    def batch_read(paths: str) -> str:
        file_list = [p.strip() for p in paths.split(",") if p.strip()]
        if not file_list:
            return "Error: No paths provided"
        results = []
        for path in file_list:
            try:
                p = validate_path(path)
                if not p.exists():
                    results.append(f"[{path}] Error: File not found")
                    continue
                content = p.read_text(encoding="utf-8", errors="replace")
                lines = content.splitlines()
                preview = content[:500] + (f"\n... ({len(content)-500} more)" if len(content) > 500 else "")
                results.append(f"[{path}] ({len(lines)} lines)\n{preview}")
            except (PermissionError, FileNotFoundError) as e:
                results.append(f"[{path}] Error: {e}")
        return f"Batch Read: {len(file_list)} files\n" + "=" * 40 + "\n\n" + "\n\n".join(results)
