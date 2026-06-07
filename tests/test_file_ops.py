import os
import tempfile
import shutil
import pytest
from pathlib import Path
from unittest.mock import patch

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.file_ops import FileOps


@pytest.fixture
def tmp_dir(monkeypatch):
    d = tempfile.mkdtemp()
    monkeypatch.setattr(os, "getcwd", lambda: d)
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def file_ops(tmp_dir):
    return FileOps(tmp_dir)


class TestInit:
    def test_default_base_dir(self, file_ops, tmp_dir):
        assert file_ops.base_dir == Path(tmp_dir).resolve()

    def test_custom_base_dir(self, tmp_dir):
        ops = FileOps(tmp_dir)
        assert ops.base_dir == Path(tmp_dir).resolve()


class TestWriteAndReadFile:
    def test_write_and_read_round_trip(self, tmp_dir):
        path = os.path.join(tmp_dir, "test.txt")
        content = "Hello, world!\nLine 2\nLine 3"
        result = FileOps.write_file(path, content)
        assert "Written" in result
        assert f"{len(content)} bytes" in result

        read_result = FileOps.read_file(path)
        assert "Hello, world!" in read_result
        assert "Line 2" in read_result
        assert "Line 3" in read_result
        assert "3 lines" in read_result

    def test_write_creates_directories(self, tmp_dir):
        path = os.path.join(tmp_dir, "sub", "dir", "file.txt")
        result = FileOps.write_file(path, "nested")
        assert "Written" in result
        assert os.path.isfile(path)

    def test_read_file_not_found(self, tmp_dir):
        result = FileOps.read_file(os.path.join(tmp_dir, "nonexistent.txt"))
        assert "Error" in result
        assert "not found" in result.lower()

    def test_read_file_with_offset_and_limit(self, tmp_dir):
        path = os.path.join(tmp_dir, "lines.txt")
        lines = "\n".join([f"line{i}" for i in range(50)])
        FileOps.write_file(path, lines)

        result = FileOps.read_file(path, offset=5, limit=5)
        assert "line5" in result
        assert "line9" in result
        assert "showing 6-10 of 50" in result

    def test_read_directory_as_file(self, tmp_dir):
        result = FileOps.read_file(tmp_dir)
        assert "Error" in result


class TestListDirectory:
    def test_list_directory(self, tmp_dir):
        FileOps.write_file(os.path.join(tmp_dir, "a.txt"), "aaa")
        FileOps.write_file(os.path.join(tmp_dir, "b.txt"), "bbbbbbbbbb")
        os.makedirs(os.path.join(tmp_dir, "subdir"))

        result = FileOps.list_directory(tmp_dir)
        assert "a.txt" in result
        assert "b.txt" in result
        assert "subdir/" in result
        assert "3 items" in result

    def test_list_directory_not_found(self, tmp_dir):
        result = FileOps.list_directory(os.path.join(tmp_dir, "nonexistent"))
        assert "Error" in result

    def test_list_directory_file_not_dir(self, tmp_dir):
        path = os.path.join(tmp_dir, "file.txt")
        FileOps.write_file(path, "content")
        result = FileOps.list_directory(path)
        assert "Error" in result


class TestFileInfo:
    def test_file_info(self, tmp_dir):
        path = os.path.join(tmp_dir, "info.txt")
        FileOps.write_file(path, "some content")

        result = FileOps.file_info(path)
        assert "Size:" in result
        assert "Modified:" in result
        assert "Is dir: False" in result

    def test_file_info_not_found(self, tmp_dir):
        result = FileOps.file_info(os.path.join(tmp_dir, "nope.txt"))
        assert "Error" in result

    def test_file_info_directory(self, tmp_dir):
        result = FileOps.file_info(tmp_dir)
        assert "Is dir: True" in result


class TestGlobSearch:
    def test_glob_search(self, tmp_dir):
        FileOps.write_file(os.path.join(tmp_dir, "a.py"), "print('a')")
        FileOps.write_file(os.path.join(tmp_dir, "b.py"), "print('b')")
        FileOps.write_file(os.path.join(tmp_dir, "c.txt"), "text")

        result = FileOps.glob_search("*.py", tmp_dir)
        assert "2" in result
        assert "a.py" in result
        assert "b.py" in result
        assert "c.txt" not in result

    def test_glob_search_no_matches(self, tmp_dir):
        FileOps.write_file(os.path.join(tmp_dir, "f.txt"), "x")
        result = FileOps.glob_search("*.xyz", tmp_dir)
        assert "No files matching" in result

    def test_glob_search_path_not_found(self, tmp_dir):
        result = FileOps.glob_search("*.py", os.path.join(tmp_dir, "nope"))
        assert "Error" in result


class TestEditFile:
    def test_edit_file(self, tmp_dir):
        path = os.path.join(tmp_dir, "edit.txt")
        FileOps.write_file(path, "Hello world\nFoo bar")

        result = FileOps.edit_file(path, "world", "Earth")
        assert "Replaced 1 occurrence" in result

        content = FileOps.read_file(path)
        assert "Hello Earth" in content
        assert "Foo bar" in content

    def test_edit_file_multiple_occurrences(self, tmp_dir):
        path = os.path.join(tmp_dir, "multi.txt")
        FileOps.write_file(path, "aaa bbb aaa ccc aaa")

        result = FileOps.edit_file(path, "aaa", "zzz")
        assert "Replaced 3 occurrence" in result

        content = FileOps.read_file(path)
        assert "zzz bbb zzz ccc zzz" in content

    def test_edit_file_not_found(self, tmp_dir):
        result = FileOps.edit_file(os.path.join(tmp_dir, "missing.txt"), "a", "b")
        assert "Error" in result

    def test_edit_file_old_string_not_found(self, tmp_dir):
        path = os.path.join(tmp_dir, "nochange.txt")
        FileOps.write_file(path, "hello")

        result = FileOps.edit_file(path, "xyz", "abc")
        assert "old_string not found" in result


class TestGrepSearch:
    def test_grep_search(self, tmp_dir):
        FileOps.write_file(os.path.join(tmp_dir, "a.txt"), "alpha\nbeta\ngamma")
        FileOps.write_file(os.path.join(tmp_dir, "b.txt"), "delta\nepsilon")

        result = FileOps.grep_search("beta", tmp_dir, "*.txt")
        assert "beta" in result
        assert "a.txt:" in result

    def test_grep_search_case_insensitive(self, tmp_dir):
        FileOps.write_file(os.path.join(tmp_dir, "f.txt"), "Hello\nhello\nHELLO")

        result = FileOps.grep_search("hello", tmp_dir, "*.txt")
        assert "(3):" in result

    def test_grep_search_no_matches(self, tmp_dir):
        FileOps.write_file(os.path.join(tmp_dir, "f.txt"), "nothing here")
        result = FileOps.grep_search("xyz", tmp_dir, "*.txt")
        assert "No matches" in result

    def test_grep_search_path_not_found(self, tmp_dir):
        result = FileOps.grep_search("test", os.path.join(tmp_dir, "nope"), "*.txt")
        assert "Error" in result
