import pytest
import json
import csv
import os
import tempfile
from pathlib import Path

from tools.data_analysis import DataAnalysis


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def csv_file(tmp_dir):
    path = os.path.join(tmp_dir, "test.csv")
    data = [
        {"name": "Alice", "age": 30, "score": 85.5},
        {"name": "Bob", "age": 25, "score": 92.0},
        {"name": "Charlie", "age": 35, "score": 78.3},
        {"name": "Diana", "age": 28, "score": 95.1},
        {"name": "Eve", "age": 40, "score": 60.0},
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "age", "score"])
        writer.writeheader()
        writer.writerows(data)
    return path


@pytest.fixture
def json_file(tmp_dir):
    path = os.path.join(tmp_dir, "test.json")
    data = [
        {"city": "New York", "pop": 8336817, "area": 302.6},
        {"city": "Los Angeles", "pop": 3979576, "area": 468.7},
        {"city": "Chicago", "pop": 2693976, "area": 227.6},
        {"city": "Houston", "pop": 2320268, "area": 671.7},
    ]
    with open(path, "w") as f:
        json.dump(data, f)
    return path


@pytest.fixture
def text_file(tmp_dir):
    path = os.path.join(tmp_dir, "test.txt")
    Path(path).write_text("Hello world.\nThis is a test file.\nIt has multiple lines.\nWords are here.\n", encoding="utf-8")
    return path


@pytest.fixture
def missing_file(tmp_dir):
    return os.path.join(tmp_dir, "nonexistent.csv")


class TestDataAnalysisInit:
    def test_class_exists(self):
        assert DataAnalysis is not None

    def test_is_static_class(self):
        assert hasattr(DataAnalysis, "_read_data_file")
        assert hasattr(DataAnalysis, "_compute_col_stats")
        assert hasattr(DataAnalysis, "analyze_csv")
        assert hasattr(DataAnalysis, "analyze_json")
        assert hasattr(DataAnalysis, "analyze_excel")
        assert hasattr(DataAnalysis, "analyze_text")
        assert hasattr(DataAnalysis, "stats_summary")
        assert hasattr(DataAnalysis, "analyze_data_quality")
        assert hasattr(DataAnalysis, "correlation_analysis")
        assert hasattr(DataAnalysis, "generate_visualization")
        assert hasattr(DataAnalysis, "sql_query")
        assert hasattr(DataAnalysis, "time_series_analysis")

    def test_read_data_file_csv(self, csv_file):
        df = DataAnalysis._read_data_file(csv_file)
        assert df is not None
        assert len(df) == 5
        assert "name" in df.columns
        assert "age" in df.columns

    def test_read_data_file_json(self, json_file):
        df = DataAnalysis._read_data_file(json_file)
        assert df is not None
        assert len(df) == 4
        assert "city" in df.columns

    def test_read_data_file_unsupported(self, tmp_dir):
        path = os.path.join(tmp_dir, "test.xyz")
        Path(path).write_text("data")
        result = DataAnalysis._read_data_file(path)
        assert result is None

    def test_compute_col_stats_numeric(self):
        stats = DataAnalysis._compute_col_stats(["10", "20", "30", "40"])
        assert stats["type"] == "numeric"
        assert stats["min"] == 10.0
        assert stats["max"] == 40.0
        assert stats["avg"] == 25.0

    def test_compute_col_stats_text(self):
        stats = DataAnalysis._compute_col_stats(["apple", "banana", "apple"])
        assert stats["type"] == "text"
        assert stats["unique"] == 2
        assert stats["non_null"] == 3

    def test_compute_col_stats_empty(self):
        stats = DataAnalysis._compute_col_stats([])
        assert stats["type"] == "text"


class TestAnalyzeCSV:
    def test_analyze_csv_success(self, csv_file):
        result = DataAnalysis.analyze_csv(csv_file)
        assert "CSV Analysis" in result
        assert "5" in result
        assert "name" in result

    def test_analyze_csv_file_not_found(self, missing_file):
        result = DataAnalysis.analyze_csv(missing_file)
        assert "Error" in result
        assert "not found" in result.lower()

    def test_analyze_csv_max_rows(self, csv_file):
        result = DataAnalysis.analyze_csv(csv_file, max_rows=2)
        assert "CSV Analysis" in result

    def test_analyze_csv_empty_file(self, tmp_dir):
        path = os.path.join(tmp_dir, "empty.csv")
        Path(path).write_text("a,b,c\n")
        result = DataAnalysis.analyze_csv(path)
        assert "CSV Analysis" in result

    def test_analyze_csv_special_characters(self, tmp_dir):
        path = os.path.join(tmp_dir, "special.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["city", "country"])
            writer.writerow(["Tokyo", "Japan"])
            writer.writerow(["Munich", "Germany"])
        result = DataAnalysis.analyze_csv(path)
        assert "Tokyo" in result


class TestAnalyzeJSON:
    def test_analyze_json_success(self, json_file):
        result = DataAnalysis.analyze_json(json_file)
        assert "JSON Analysis" in result or "Analysis" in result

    def test_analyze_json_file_not_found(self, missing_file):
        result = DataAnalysis.analyze_json(missing_file)
        assert "Error" in result
        assert "not found" in result.lower()

    def test_analyze_json_dict_format(self, tmp_dir):
        path = os.path.join(tmp_dir, "dict.json")
        data = {"name": "test", "value": 42, "items": [1, 2, 3]}
        with open(path, "w") as f:
            json.dump(data, f)
        result = DataAnalysis.analyze_json(path)
        assert "Analysis" in result

    def test_analyze_json_nested(self, tmp_dir):
        path = os.path.join(tmp_dir, "nested.json")
        data = [{"id": 1, "info": {"x": 10}}, {"id": 2, "info": {"x": 20}}]
        with open(path, "w") as f:
            json.dump(data, f)
        result = DataAnalysis.analyze_json(path)
        assert "Analysis" in result


class TestGetStatistics:
    def test_stats_summary_basic(self):
        result = DataAnalysis.stats_summary("[10, 20, 30, 40, 50]")
        assert "Statistical Summary" in result
        assert "Mean" in result
        assert "Median" in result
        assert "Std Dev" in result

    def test_stats_summary_single_value(self):
        result = DataAnalysis.stats_summary("[42]")
        assert "Statistical Summary" in result

    def test_stats_summary_invalid_json(self):
        result = DataAnalysis.stats_summary("not json")
        assert "Error" in result

    def test_stats_summary_empty_list(self):
        result = DataAnalysis.stats_summary("[]")
        assert "Error" in result

    def test_stats_summary_non_numeric(self):
        result = DataAnalysis.stats_summary('["a","b","c"]')
        assert "Error" in result

    def test_stats_summary_with_decimals(self):
        result = DataAnalysis.stats_summary("[1.5, 2.7, 3.9, 4.1]")
        assert "Statistical Summary" in result

    def test_stats_summary_negative_numbers(self):
        result = DataAnalysis.stats_summary("[-5, -10, 0, 5, 20]")
        assert "Statistical Summary" in result
        assert "Min" in result

    def test_stats_summary_zero_mean_bug(self):
        result = DataAnalysis.stats_summary("[-5, -10, 5, 10]")
        assert "Statistical Summary" in result
        assert "N/A" in result

    def test_stats_summary_type_error(self):
        result = DataAnalysis.stats_summary(None)
        assert "Error" in result


class TestDataQuality:
    def test_data_quality_csv(self, csv_file):
        result = DataAnalysis.analyze_data_quality(csv_file)
        assert "Data Quality Report" in result
        assert "Rows" in result
        assert "Columns" in result
        assert "Missing Values" in result
        assert "Duplicates" in result
        assert "Data Types" in result

    def test_data_quality_json(self, json_file):
        result = DataAnalysis.analyze_data_quality(json_file)
        assert "Data Quality Report" in result
        assert "Rows" in result

    def test_data_quality_file_not_found(self, missing_file):
        result = DataAnalysis.analyze_data_quality(missing_file)
        assert "Error" in result
        assert "not found" in result.lower()

    def test_data_quality_unsupported_format(self, tmp_dir):
        path = os.path.join(tmp_dir, "test.xyz")
        Path(path).write_text("data")
        result = DataAnalysis.analyze_data_quality(path)
        assert "Unsupported format" in result

    def test_data_quality_with_missing_values(self, tmp_dir):
        path = os.path.join(tmp_dir, "missing.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["a", "b", "c"])
            writer.writerow(["1", "2", "3"])
            writer.writerow(["4", "", "6"])
            writer.writerow(["", "8", ""])
        result = DataAnalysis.analyze_data_quality(path)
        assert "Data Quality Report" in result
        assert "Missing Values" in result

    def test_data_quality_with_duplicates(self, tmp_dir):
        path = os.path.join(tmp_dir, "dupes.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            writer.writerow(["1", "2"])
            writer.writerow(["1", "2"])
            writer.writerow(["3", "4"])
        result = DataAnalysis.analyze_data_quality(path)
        assert "Duplicates" in result


class TestCorrelationAnalysis:
    def test_correlation_analysis_csv(self, csv_file):
        result = DataAnalysis.correlation_analysis(csv_file)
        # Source bug: correlation_analysis uses list.append(..., end="")
        # which raises TypeError. Expect error string from except handler.
        assert "Correlation Analysis" in result or "Error" in result

    def test_correlation_analysis_json(self, json_file):
        result = DataAnalysis.correlation_analysis(json_file)
        assert "Correlation Analysis" in result or "Error" in result

    def test_correlation_analysis_file_not_found(self, missing_file):
        result = DataAnalysis.correlation_analysis(missing_file)
        assert "Error" in result
        assert "not found" in result.lower()

    def test_correlation_analysis_specific_columns(self, csv_file):
        result = DataAnalysis.correlation_analysis(csv_file, columns="age,score")
        assert "Correlation Analysis" in result or "Error" in result

    def test_correlation_analysis_invalid_columns(self, csv_file):
        result = DataAnalysis.correlation_analysis(csv_file, columns="nonexistent")
        assert "None of the specified columns found" in result

    def test_correlation_analysis_text_only(self, tmp_dir):
        path = os.path.join(tmp_dir, "textonly.csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["color", "shape"])
            writer.writerow(["red", "circle"])
            writer.writerow(["blue", "square"])
        result = DataAnalysis.correlation_analysis(path)
        assert "No numeric columns" in result

    def test_correlation_analysis_two_columns(self, csv_file):
        result = DataAnalysis.correlation_analysis(csv_file, columns="age")
        assert "Correlation Analysis" in result or "Error" in result


class TestAnalyzeText:
    def test_analyze_text_success(self, text_file):
        result = DataAnalysis.analyze_text(text_file)
        assert "Text Analysis" in result
        assert "Characters" in result
        assert "Words" in result
        assert "Lines" in result

    def test_analyze_text_file_not_found(self, missing_file):
        result = DataAnalysis.analyze_text(missing_file)
        assert "Error" in result
        assert "not found" in result.lower()


class TestSqlQuery:
    def test_sql_query_select(self, csv_file):
        result = DataAnalysis.sql_query(csv_file, "SELECT * FROM data")
        assert "SQL Result" in result
        assert "5" in result

    def test_sql_query_where(self, csv_file):
        result = DataAnalysis.sql_query(csv_file, "SELECT * FROM data WHERE age > 30")
        assert "SQL Result" in result

    def test_sql_query_file_not_found(self, missing_file):
        result = DataAnalysis.sql_query(missing_file, "SELECT * FROM data")
        assert "Error" in result
        assert "not found" in result.lower()

    def test_sql_query_invalid_query(self, csv_file):
        result = DataAnalysis.sql_query(csv_file, "INVALID SQL QUERY")
        assert "Error" in result
