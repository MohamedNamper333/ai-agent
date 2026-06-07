"""Advanced security scanner module for AI Agent"""
import re
from typing import Dict, List


class SecurityScanner:
    def __init__(self):
        self.patterns = {
            "sql_injection": {
                "patterns": [
                    r"SELECT\s+.*FROM\s+.*WHERE\s+.*\+",
                    r"INSERT\s+INTO\s+.*VALUES\s+.*\+",
                    r"UPDATE\s+.*SET\s+.*WHERE\s+.*\+",
                    r"DELETE\s+FROM\s+.*WHERE\s+.*\+",
                    r"execute\s*\(.*['\"].*['\"]",
                ],
                "risk_level": "high",
                "description": "Potential SQL injection vulnerability"
            },
            "xss": {
                "patterns": [
                    r"innerHTML\s*=\s*",
                    r"document\.write\s*\(",
                    r"\.html\s*\(",
                    r"eval\s*\(",
                ],
                "risk_level": "high",
                "description": "Potential XSS vulnerability"
            },
            "hardcoded_secrets": {
                "patterns": [
                    r"password\s*=\s*['\"][^'\"]+['\"]",
                    r"api_key\s*=\s*['\"][^'\"]+['\"]",
                    r"secret\s*=\s*['\"][^'\"]+['\"]",
                    r"token\s*=\s*['\"][^'\"]+['\"]",
                    r"sk-[a-zA-Z0-9]{20,}",
                ],
                "risk_level": "high",
                "description": "Hardcoded secret or credential"
            },
            "insecure_random": {
                "patterns": [
                    r"random\.random\s*\(\)",
                    r"Math\.random\s*\(\)",
                ],
                "risk_level": "medium",
                "description": "Insecure random number generation"
            },
            "eval_usage": {
                "patterns": [
                    r"eval\s*\(",
                    r"exec\s*\(",
                    r"compile\s*\(.*['\"]exec['\"]",
                ],
                "risk_level": "high",
                "description": "Dynamic code execution (potential code injection)"
            },
            "path_traversal": {
                "patterns": [
                    r"\.\.\/",
                    r"\.\.\\",
                    r"open\s*\(.*\.\.",
                ],
                "risk_level": "medium",
                "description": "Potential path traversal"
            },
            "insecure_deserialization": {
                "patterns": [
                    r"pickle\.loads?\s*\(",
                    r"yaml\.load\s*\(",
                    r"marshal\.loads?\s*\(",
                ],
                "risk_level": "high",
                "description": "Insecure deserialization"
            },
            "debug_code": {
                "patterns": [
                    r"print\s*\(.*debug",
                    r"console\.log\s*\(",
                    r"import\s+pdb",
                    r"breakpoint\s*\(\)",
                ],
                "risk_level": "low",
                "description": "Debug code in production"
            },
        }

        self.recommendations = {
            "high": [
                "Use parameterized queries for database operations",
                "Sanitize and validate all user inputs",
                "Use environment variables for secrets",
                "Avoid eval() and exec() - use safer alternatives",
                "Use secure deserialization libraries",
            ],
            "medium": [
                "Use cryptographically secure random generators",
                "Validate file paths to prevent traversal",
                "Use context managers for file operations",
                "Add input length limits",
            ],
            "low": [
                "Remove debug statements before production",
                "Use proper logging instead of print",
                "Configure logging levels appropriately",
            ],
        }

    def scan_code(self, code: str) -> Dict:
        """Scan code for security issues"""
        issues = []
        risk_scores = {"low": 0, "medium": 0, "high": 0}

        for category, config in self.patterns.items():
            for pattern in config["patterns"]:
                if re.search(pattern, code, re.IGNORECASE):
                    issues.append(f"{config['description']} ({category})")
                    risk_scores[config["risk_level"]] += 1
                    break

        if risk_scores["high"] > 0:
            risk_level = "high"
        elif risk_scores["medium"] > 0:
            risk_level = "medium"
        elif risk_scores["low"] > 0:
            risk_level = "low"
        else:
            risk_level = "low"

        return {
            "risk_level": risk_level,
            "issues": issues,
            "issue_count": len(issues),
            "risk_scores": risk_scores
        }

    def scan_file(self, file_path: str) -> Dict:
        """Scan a file for security issues"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()

            result = self.scan_code(code)
            result["file_path"] = file_path
            return result
        except Exception as e:
            return {
                "risk_level": "unknown",
                "issues": [f"Failed to scan file: {str(e)}"],
                "issue_count": 0,
                "file_path": file_path
            }

    def get_recommendations(self, risk_level: str) -> List[str]:
        """Get security recommendations based on risk level"""
        if risk_level == "high":
            return self.recommendations["high"] + self.recommendations["medium"]
        elif risk_level == "medium":
            return self.recommendations["medium"]
        else:
            return self.recommendations["low"]

    def generate_report(self, scan_result: Dict) -> str:
        """Generate a human-readable security report"""
        report_lines = [
            "Security Report",
            "================",
            f"Risk Level: {scan_result['risk_level'].upper()}",
            f"Issues Found: {scan_result['issue_count']}",
            ""
        ]

        if scan_result["issues"]:
            report_lines.append("Issues:")
            for i, issue in enumerate(scan_result["issues"], 1):
                report_lines.append(f"  {i}. {issue}")
            report_lines.append("")

        recommendations = self.get_recommendations(scan_result["risk_level"])
        report_lines.append("Recommendations:")
        for rec in recommendations:
            report_lines.append(f"  - {rec}")

        return "\n".join(report_lines)
