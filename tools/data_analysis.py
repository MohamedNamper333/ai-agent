import csv
import json
import os
import io
import base64
from pathlib import Path


class DataAnalysis:
    @staticmethod
    def analyze_csv(file_path: str, max_rows: int = 50) -> str:
        p = Path(file_path)
        if not p.exists():
            return f"Error: File not found: {file_path}"

        try:
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    return "Error: No columns found in CSV"

                rows = []
                for i, row in enumerate(reader):
                    if i >= max_rows:
                        break
                    rows.append(row)

                col_types = {}
                col_stats = {}
                for col in reader.fieldnames:
                    values = [r.get(col, "") for r in rows]
                    nums = []
                    for v in values:
                        try:
                            nums.append(float(v) if v else 0)
                        except (ValueError, TypeError):
                            pass

                    if nums:
                        col_types[col] = "numeric"
                        col_stats[col] = {
                            "min": min(nums),
                            "max": max(nums),
                            "avg": round(sum(nums) / len(nums), 3),
                            "non_null": len(nums),
                        }
                    else:
                        non_empty = [v for v in values if v.strip()]
                        col_types[col] = "text"
                        col_stats[col] = {"unique": len(set(values)), "non_null": len(non_empty)}

                result = [
                    f"CSV Analysis: {file_path}",
                    f"Columns ({len(reader.fieldnames)}): {', '.join(reader.fieldnames)}",
                    f"Rows shown: {len(rows)}",
                    "",
                    "Column Summary:",
                ]
                for col in reader.fieldnames:
                    t = col_types.get(col, "unknown")
                    s = col_stats.get(col, {})
                    result.append(f"  {col} ({t}): {s}")

                result.append("")
                result.append("First 5 rows:")
                for row in rows[:5]:
                    result.append(f"  {row}")

                return "\n".join(result)

        except Exception as e:
            return f"Error analyzing CSV: {e}"

    @staticmethod
    def analyze_json(file_path: str) -> str:
        p = Path(file_path)
        if not p.exists():
            return f"Error: File not found: {file_path}"

        try:
            data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
            import json as json_module

            formatted = json_module.dumps(data, indent=2, ensure_ascii=False)
            type_name = type(data).__name__

            summary = f"JSON Analysis: {file_path}\n"
            summary += f"Type: {type_name}\n"

            if isinstance(data, dict):
                summary += f"Keys ({len(data)}): {list(data.keys())[:20]}\n"
            elif isinstance(data, list):
                summary += f"Length: {len(data)}\n"
                if data:
                    summary += f"Element type: {type(data[0]).__name__}\n"
                    if isinstance(data[0], dict):
                        summary += f"Keys in first element: {list(data[0].keys())[:20]}\n"

            if len(formatted) > 3000:
                formatted = formatted[:3000] + "\n... (truncated)"
            summary += f"\n{formatted}"
            return summary

        except Exception as e:
            return f"Error analyzing JSON: {e}"

    @staticmethod
    def analyze_text(file_path: str) -> str:
        p = Path(file_path)
        if not p.exists():
            return f"Error: File not found: {file_path}"

        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            lines = text.splitlines()
            words = text.split()

            char_freq = {}
            for c in text.lower():
                if c.isalpha():
                    char_freq[c] = char_freq.get(c, 0) + 1
            top_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)[:10]

            return (
                f"Text Analysis: {file_path}\n"
                f"  Characters: {len(text):,}\n"
                f"  Words: {len(words):,}\n"
                f"  Lines: {len(lines):,}\n"
                f"  Avg word length: {sum(len(w) for w in words)/max(len(words),1):.1f}\n"
                f"  Most common letters: {', '.join(f'{c}={n}' for c,n in top_chars)}\n"
            )
        except Exception as e:
            return f"Error analyzing text: {e}"

    @staticmethod
    def stats_summary(data_json: str) -> str:
        try:
            data = json.loads(data_json)
        except (json.JSONDecodeError, TypeError):
            return "Error: Provide data as a JSON string (array of numbers)"

        if not isinstance(data, list) or not data:
            return "Error: Data must be a non-empty array"

        nums = []
        for item in data:
            try:
                nums.append(float(item))
            except (ValueError, TypeError):
                continue

        if not nums:
            return "Error: No numeric values found in data"

        sorted_nums = sorted(nums)
        n = len(sorted_nums)
        mean = sum(sorted_nums) / n
        variance = sum((x - mean) ** 2 for x in sorted_nums) / n
        std_dev = variance ** 0.5
        median = sorted_nums[n // 2] if n % 2 else (sorted_nums[n // 2 - 1] + sorted_nums[n // 2]) / 2

        quartile_1 = sorted_nums[n // 4]
        quartile_3 = sorted_nums[3 * n // 4]

        return (
            f"Statistical Summary (n={n}):\n"
            f"  Min: {min(nums):.4f}\n"
            f"  Max: {max(nums):.4f}\n"
            f"  Mean: {mean:.4f}\n"
            f"  Median: {median:.4f}\n"
            f"  Std Dev: {std_dev:.4f}\n"
            f"  Q1: {quartile_1:.4f}\n"
            f"  Q3: {quartile_3:.4f}\n"
            f"  Range: {max(nums) - min(nums):.4f}\n"
            f"  Sum: {sum(nums):.4f}\n"
        )
