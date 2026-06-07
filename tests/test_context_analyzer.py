"""Tests for core.context_analyzer module"""
import pytest
from core.context_analyzer import ContextAnalyzer


@pytest.fixture
def analyzer():
    return ContextAnalyzer()


class TestContextAnalyzerInit:
    def test_init(self, analyzer):
        assert analyzer is not None

    def test_has_intent_patterns(self, analyzer):
        assert hasattr(analyzer, 'intent_patterns')
        assert len(analyzer.intent_patterns) > 0

    def test_has_entity_patterns(self, analyzer):
        assert hasattr(analyzer, 'entity_patterns')
        assert len(analyzer.entity_patterns) > 0


class TestAnalyzeIntent:
    def test_code_generation(self, analyzer):
        result = analyzer.analyze_intent("Write a Python function to sort a list")
        assert result["intent"] == "code_generation"
        assert result["confidence"] > 0

    def test_code_analysis(self, analyzer):
        result = analyzer.analyze_intent("Analyze the performance of this code")
        assert result["intent"] == "code_analysis"
        assert result["confidence"] > 0

    def test_web_search(self, analyzer):
        result = analyzer.analyze_intent("Search for latest AI news")
        assert result["intent"] == "web_search"
        assert result["confidence"] > 0

    def test_file_operation(self, analyzer):
        result = analyzer.analyze_intent("Read the file data.csv")
        assert result["intent"] == "file_operation"
        assert result["confidence"] > 0

    def test_data_analysis(self, analyzer):
        result = analyzer.analyze_intent("Analyze the data and show statistics")
        assert result["intent"] == "data_analysis"
        assert result["confidence"] > 0

    def test_explanation(self, analyzer):
        result = analyzer.analyze_intent("Explain how this algorithm works")
        assert result["intent"] == "explanation"
        assert result["confidence"] > 0

    def test_general(self, analyzer):
        result = analyzer.analyze_intent("hello")
        assert result["intent"] == "general"


class TestExtractEntities:
    def test_file_path(self, analyzer):
        result = analyzer.extract_entities("Read the file data.csv and analyze it")
        assert "file_path" in result
        assert "data.csv" in result["file_path"]

    def test_programming_language(self, analyzer):
        result = analyzer.extract_entities("Write a Python function")
        assert "programming_language" in result
        assert "python" in [x.lower() for x in result["programming_language"]]

    def test_framework(self, analyzer):
        result = analyzer.extract_entities("Build a Flask web app")
        assert "framework" in result
        assert "flask" in [x.lower() for x in result["framework"]]

    def test_concept(self, analyzer):
        result = analyzer.extract_entities("Implement a binary search algorithm")
        assert "concept" in result
        assert "algorithm" in result["concept"]

    def test_no_entities(self, analyzer):
        result = analyzer.extract_entities("hello world")
        assert len(result) == 0


class TestRelevanceScore:
    def test_high_relevance(self, analyzer):
        score = analyzer.calculate_relevance_score(
            "Python programming",
            "This document discusses Python best practices"
        )
        assert score > 0.5

    def test_low_relevance(self, analyzer):
        score = analyzer.calculate_relevance_score(
            "Java programming",
            "This document discusses Python best practices"
        )
        assert score < 0.5

    def test_empty_query(self, analyzer):
        score = analyzer.calculate_relevance_score("", "some context")
        assert score == 0.0


class TestSuggestedTools:
    def test_code_generation_tools(self, analyzer):
        tools = analyzer.get_suggested_tools("code_generation", {})
        assert "run_code" in tools

    def test_web_search_tools(self, analyzer):
        tools = analyzer.get_suggested_tools("web_search", {})
        assert "search_web" in tools

    def test_file_operation_tools(self, analyzer):
        tools = analyzer.get_suggested_tools("file_operation", {})
        assert "read_file" in tools

    def test_with_entities(self, analyzer):
        entities = {"programming_language": ["python"]}
        tools = analyzer.get_suggested_tools("code_generation", entities)
        assert "run_code" in tools
