"""Tests for core.security_scanner module"""
import pytest
import tempfile
import os
from core.security_scanner import SecurityScanner


@pytest.fixture
def scanner():
    return SecurityScanner()


class TestSecurityScannerInit:
    def test_init(self, scanner):
        assert scanner is not None

    def test_has_patterns(self, scanner):
        assert hasattr(scanner, 'patterns')
        assert len(scanner.patterns) > 0

    def test_has_recommendations(self, scanner):
        assert hasattr(scanner, 'recommendations')
        assert "high" in scanner.recommendations


class TestScanCode:
    def test_safe_code(self, scanner):
        code = "def add(a, b): return a + b"
        result = scanner.scan_code(code)
        assert result["risk_level"] == "low"
        assert result["issue_count"] == 0

    def test_sql_injection(self, scanner):
        code = 'query = "SELECT * FROM users WHERE id = " + user_input'
        result = scanner.scan_code(code)
        assert result["risk_level"] == "high"
        assert result["issue_count"] > 0

    def test_xss(self, scanner):
        code = 'element.innerHTML = userInput'
        result = scanner.scan_code(code)
        assert result["risk_level"] == "high"

    def test_hardcoded_secrets(self, scanner):
        code = 'password = "secret123"\napi_key = "sk-12345678901234567890"'
        result = scanner.scan_code(code)
        assert result["risk_level"] == "high"

    def test_eval_usage(self, scanner):
        code = 'result = eval(user_input)'
        result = scanner.scan_code(code)
        assert result["risk_level"] == "high"

    def test_debug_code(self, scanner):
        code = 'print("debug:", variable)'
        result = scanner.scan_code(code)
        assert result["risk_level"] == "low"

    def test_insecure_random(self, scanner):
        code = 'value = random.random()'
        result = scanner.scan_code(code)
        assert result["risk_level"] == "medium"


class TestScanFile:
    def test_scan_safe_file(self, scanner, tmp_path):
        safe_file = tmp_path / "safe.py"
        safe_file.write_text("def hello(): return 'world'")
        result = scanner.scan_file(str(safe_file))
        assert result["risk_level"] == "low"

    def test_scan_unsafe_file(self, scanner, tmp_path):
        unsafe_file = tmp_path / "unsafe.py"
        unsafe_file.write_text('password = "secret123"')
        result = scanner.scan_file(str(unsafe_file))
        assert result["risk_level"] == "high"

    def test_scan_nonexistent_file(self, scanner):
        result = scanner.scan_file("nonexistent.py")
        assert result["risk_level"] == "unknown"


class TestGetRecommendations:
    def test_high_risk(self, scanner):
        recs = scanner.get_recommendations("high")
        assert len(recs) > 0
        assert any("parameterized" in r.lower() for r in recs)

    def test_medium_risk(self, scanner):
        recs = scanner.get_recommendations("medium")
        assert len(recs) > 0

    def test_low_risk(self, scanner):
        recs = scanner.get_recommendations("low")
        assert len(recs) > 0


class TestGenerateReport:
    def test_report_generation(self, scanner):
        scan_result = {
            "risk_level": "high",
            "issues": ["SQL injection vulnerability"],
            "issue_count": 1,
            "risk_scores": {"low": 0, "medium": 0, "high": 1}
        }
        report = scanner.generate_report(scan_result)
        assert "HIGH" in report
        assert "SQL injection" in report
        assert "Recommendations:" in report
