import os
import tempfile
import textwrap
import pytest
from tools.code_analysis import CodeAnalysis


@pytest.fixture
def tmp_project(tmp_path):
    """Create a temporary Python project with various patterns."""
    src = tmp_path / "src"
    src.mkdir()
    pkg = src / "mypkg"
    pkg.mkdir()

    (pkg / "__init__.py").write_text("# init\n", encoding="utf-8")

    (pkg / "utils.py").write_text(textwrap.dedent("""\
        import os
        import sys
        import json
        from pathlib import Path

        class Helper:
            def __init__(self, root):
                self.root = root

            def list_files(self):
                result = []
                for f in os.listdir(self.root):
                    if f.endswith('.py'):
                        result.append(f)
                return result

            def read_json(self, path):
                with open(path) as fh:
                    return json.load(fh)

        def compute(x, y):
            if x > 0:
                if y > 0:
                    return x + y
                else:
                    return x - y
            elif x < 0:
                return -x
            return 0

        # TODO: add docstrings
    """), encoding="utf-8")

    (pkg / "security_test.py").write_text(textwrap.dedent("""\
        import os
        import pickle
        import subprocess
        import yaml

        def dangerous_load(data):
            return pickle.loads(data)

        def run_cmd(cmd):
            return subprocess.call(cmd)

        def load_yaml(text):
            return yaml.load(text)

        def exec_code(code):
            exec(code)

        eval("1+1")
    """), encoding="utf-8")

    (pkg / "smells.py").write_text(textwrap.dedent("""\
        import *

        def bad_func():
            try:
                pass
            except:
                pass
            print("debug")

        # FIXME: broken
        # HACK: workaround
    """), encoding="utf-8")

    (src / "standalone.py").write_text(textwrap.dedent("""\
        import os
        from mypkg.utils import Helper

        def main():
            h = Helper(os.getcwd())
            print(h.list_files())

        if __name__ == "__main__":
            main()
    """), encoding="utf-8")

    (tmp_path / "README.md").write_text("# test project\n", encoding="utf-8")

    return tmp_path


class TestCodeAnalysisInit:
    def test_class_exists(self):
        assert hasattr(CodeAnalysis, "SECURITY_PATTERNS")
        assert hasattr(CodeAnalysis, "CODE_SMELL_PATTERNS")

    def test_security_patterns_non_empty(self):
        assert len(CodeAnalysis.SECURITY_PATTERNS) > 0
        for pattern, desc, sev in CodeAnalysis.SECURITY_PATTERNS:
            assert isinstance(pattern, str)
            assert isinstance(desc, str)
            assert sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO")

    def test_code_smell_patterns_non_empty(self):
        assert len(CodeAnalysis.CODE_SMELL_PATTERNS) > 0
        for pattern, desc, sev in CodeAnalysis.CODE_SMELL_PATTERNS:
            assert isinstance(pattern, str)
            assert isinstance(desc, str)
            assert sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO")

    def test_static_methods_exist(self):
        static_methods = [
            "scan_project", "review_code", "analyze_imports",
            "complexity_metrics", "analyze_security", "code_refactor",
            "dependency_graph", "analyze_code_quality", "generate_test",
        ]
        for method in static_methods:
            assert hasattr(CodeAnalysis, method), f"Missing method: {method}"
            assert callable(getattr(CodeAnalysis, method))


class TestScanProject:
    def test_scan_project_with_temp_project(self, tmp_project):
        result = CodeAnalysis.scan_project(str(tmp_project))
        assert "Project:" in result
        assert "Python files:" in result
        assert "Python lines:" in result
        assert "File types:" in result
        assert "Total files:" in result

    def test_scan_project_counts_python_files(self, tmp_project):
        result = CodeAnalysis.scan_project(str(tmp_project))
        assert "Python files: 5" in result

    def test_scan_project_nonexistent_path(self, tmp_path):
        result = CodeAnalysis.scan_project(str(tmp_path / "nope"))
        assert "Error" in result

    def test_scan_project_empty_dir(self, tmp_path):
        result = CodeAnalysis.scan_project(str(tmp_path))
        assert "Python files: 0" in result
        assert "Total files: 0" in result

    def test_scan_project_max_files_limit(self, tmp_project):
        result = CodeAnalysis.scan_project(str(tmp_project), max_files=1)
        assert "Python files:" in result


class TestReviewCode:
    def test_review_basic_file(self, tmp_project):
        file_path = str(tmp_project / "src" / "mypkg" / "utils.py")
        result = CodeAnalysis.review_code(file_path)
        assert "## Code Review:" in result
        assert "### Statistics" in result
        assert "Total lines:" in result
        assert "Functions:" in result
        assert "Classes:" in result
        assert "### Quality Score:" in result

    def test_review_nonexistent_file(self):
        result = CodeAnalysis.review_code("nonexistent.py")
        assert "Error" in result

    def test_review_security_issues(self, tmp_project):
        file_path = str(tmp_project / "src" / "mypkg" / "security_test.py")
        result = CodeAnalysis.review_code(file_path)
        assert "Security Issues" in result
        assert "CRITICAL" in result

    def test_review_code_smells(self, tmp_project):
        file_path = str(tmp_project / "src" / "mypkg" / "smells.py")
        result = CodeAnalysis.review_code(file_path)
        assert "Code Smells" in result
        assert "Bare except" in result

    def test_review_suggestions_present(self, tmp_project):
        file_path = str(tmp_project / "src" / "mypkg" / "security_test.py")
        result = CodeAnalysis.review_code(file_path)
        assert "### Suggestions" in result


class TestAnalyzeImports:
    def test_analyze_imports_basic(self, tmp_project):
        file_path = str(tmp_project / "src" / "mypkg" / "utils.py")
        result = CodeAnalysis.analyze_imports(file_path)
        assert "## Import Analysis:" in result
        assert "Total:" in result
        assert "Standard Library" in result

    def test_analyze_imports_nonexistent_file(self):
        result = CodeAnalysis.analyze_imports("no_file.py")
        assert "Error" in result

    def test_analyze_imports_third_party(self, tmp_project):
        file_path = str(tmp_project / "src" / "mypkg" / "security_test.py")
        result = CodeAnalysis.analyze_imports(file_path)
        assert "Third Party" in result

    def test_analyze_imports_local(self, tmp_project):
        file_path = str(tmp_project / "src" / "standalone.py")
        result = CodeAnalysis.analyze_imports(file_path)
        assert "mypkg" in result

    def test_analyze_imports_syntax_error_fallback(self, tmp_path):
        bad_file = tmp_path / "bad.py"
        bad_file.write_text("import os\nimport json\n???bad\n", encoding="utf-8")
        result = CodeAnalysis.analyze_imports(str(bad_file))
        assert "## Import Analysis:" in result


class TestComplexityMetrics:
    def test_complexity_basic(self, tmp_project):
        file_path = str(tmp_project / "src" / "mypkg" / "utils.py")
        result = CodeAnalysis.complexity_metrics(file_path)
        assert "## Complexity Metrics:" in result
        assert "### Raw Metrics" in result
        assert "### Quality Assessment" in result
        assert "Score:" in result
        assert "Cyclomatic Complexity:" in result

    def test_complexity_nonexistent_file(self):
        result = CodeAnalysis.complexity_metrics("no_file.py")
        assert "Error" in result

    def test_complexity_function_breakdown(self, tmp_project):
        file_path = str(tmp_project / "src" / "mypkg" / "utils.py")
        result = CodeAnalysis.complexity_metrics(file_path)
        assert "Function Complexity Breakdown" in result
        assert "compute" in result
        assert "Helper" in result or "list_files" in result or "read_json" in result

    def test_complexity_empty_file(self, tmp_path):
        f = tmp_path / "empty.py"
        f.write_text("", encoding="utf-8")
        result = CodeAnalysis.complexity_metrics(str(f))
        assert "Score:" in result
        assert "Lines: 0" in result

    def test_complexity_non_python_file(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("hello world\n", encoding="utf-8")
        result = CodeAnalysis.complexity_metrics(str(f))
        assert "Lines: 1" in result


class TestSecurityScan:
    def test_security_scan_clean(self, tmp_project):
        file_path = str(tmp_project / "src" / "mypkg" / "utils.py")
        result = CodeAnalysis.analyze_security(file_path)
        assert "## Security Scan:" in result
        assert "No security issues detected" in result

    def test_security_scan_findings(self, tmp_project):
        file_path = str(tmp_project / "src" / "mypkg" / "security_test.py")
        result = CodeAnalysis.analyze_security(file_path)
        assert "## Security Scan:" in result
        assert "Critical" in result
        assert "Overall Risk:" in result

    def test_security_scan_nonexistent_file(self):
        result = CodeAnalysis.analyze_security("no_file.py")
        assert "Error" in result

    def test_security_scan_risk_levels(self, tmp_path):
        f = tmp_path / "high_risk.py"
        f.write_text(textwrap.dedent("""\
            import subprocess
            import pickle
            import yaml
            def bad(): return pickle.loads(b'')
            yaml.load("x")
            subprocess.call("ls")
        """), encoding="utf-8")
        result = CodeAnalysis.analyze_security(str(f))
        assert "Overall Risk:" in result

    def test_security_scan_low_risk(self, tmp_path):
        f = tmp_path / "low_risk.py"
        f.write_text("print('hello')\n", encoding="utf-8")
        result = CodeAnalysis.analyze_security(str(f))
        assert "No security issues detected" in result


class TestCodeRefactor:
    def test_refactor_basic(self, tmp_project):
        file_path = str(tmp_project / "src" / "mypkg" / "utils.py")
        result = CodeAnalysis.code_refactor(file_path)
        assert "## Refactoring Analysis:" in result
        assert "potential improvements" in result

    def test_refactor_with_instructions(self, tmp_project):
        file_path = str(tmp_project / "src" / "mypkg" / "utils.py")
        result = CodeAnalysis.code_refactor(file_path, "Improve security")
        assert "Improve security" in result

    def test_refactor_nonexistent_file(self):
        result = CodeAnalysis.code_refactor("no_file.py")
        assert "Error" in result

    def test_refactor_detects_smells(self, tmp_project):
        file_path = str(tmp_project / "src" / "mypkg" / "smells.py")
        result = CodeAnalysis.code_refactor(file_path)
        assert "bare except" in result.lower() or "wildcard" in result.lower() or "print" in result.lower()


class TestDependencyGraph:
    def test_dependency_graph_basic(self, tmp_project):
        result = CodeAnalysis.dependency_graph(str(tmp_project / "src"))
        assert "## Dependency Graph:" in result
        assert "Total modules:" in result

    def test_dependency_graph_nonexistent(self, tmp_path):
        result = CodeAnalysis.dependency_graph(str(tmp_path / "nope"))
        assert "Error" in result

    def test_dependency_graph_detects_deps(self, tmp_project):
        pkg = tmp_project / "src" / "mypkg"
        (pkg / "mod_a.py").write_text("x = 1\n", encoding="utf-8")
        (pkg / "mod_b.py").write_text(textwrap.dedent("""\
            import mod_a
            print(mod_a.x)
        """), encoding="utf-8")
        result = CodeAnalysis.dependency_graph(str(tmp_project / "src"))
        assert "Total modules:" in result
        assert "mod_b" in result


class TestGenerateTest:
    def test_generate_test_basic(self, tmp_project):
        file_path = str(tmp_project / "src" / "mypkg" / "utils.py")
        result = CodeAnalysis.generate_test(file_path)
        assert "import pytest" in result
        assert "class Test" in result
        assert "def test_" in result

    def test_generate_test_specific_function(self, tmp_project):
        file_path = str(tmp_project / "src" / "mypkg" / "utils.py")
        result = CodeAnalysis.generate_test(file_path, "compute")
        assert "test_compute" in result

    def test_generate_test_nonexistent_file(self):
        result = CodeAnalysis.generate_test("no_file.py")
        assert "Error" in result

    def test_generate_test_function_not_found(self, tmp_project):
        file_path = str(tmp_project / "src" / "mypkg" / "utils.py")
        result = CodeAnalysis.generate_test(file_path, "nonexistent_func")
        assert "not found" in result

    def test_generate_test_syntax_error_file(self, tmp_path):
        f = tmp_path / "broken.py"
        f.write_text("def foo(:\n", encoding="utf-8")
        result = CodeAnalysis.generate_test(str(f))
        assert "Error" in result


class TestAnalyzeCodeQuality:
    def test_quality_basic(self, tmp_project):
        file_path = str(tmp_project / "src" / "mypkg" / "utils.py")
        result = CodeAnalysis.analyze_code_quality(file_path)
        assert "## Code Quality Report:" in result
        assert "Overall Quality:" in result
        assert "Metrics Summary" in result
        assert "Issues Found" in result
        assert "Recommendations" in result

    def test_quality_nonexistent_file(self):
        result = CodeAnalysis.analyze_code_quality("no_file.py")
        assert "Error" in result

    def test_quality_non_python(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("hello\n", encoding="utf-8")
        result = CodeAnalysis.analyze_code_quality(str(f))
        assert "only supports Python" in result
