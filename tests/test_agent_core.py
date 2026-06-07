import pytest
import time
from unittest.mock import patch, MagicMock
from core.agent import Agent, ToolCall, PlanStep, ExecutionPlan, TaskStatus

AGENT_PATCHES = [
    patch.object(Agent, "_init_rag"),
    patch.object(Agent, "_start_scheduler"),
    patch.object(Agent, "_load_plugins"),
    patch("core.agent.get_cache_manager"),
    patch("core.agent.ContextManager"),
    patch("core.agent.ToolRegistry"),
    patch("core.agent.ConversationMemory"),
    patch("core.agent.LLMRouter"),
    patch("core.agent.Telemetry"),
    patch("core.agent.CoTEngine"),
]


def make_agent():
    patches = [p.start() for p in AGENT_PATCHES]
    mock_cache = patches[3]
    mock_cache.return_value.get_cache.return_value = MagicMock()
    try:
        return Agent()
    finally:
        for p in AGENT_PATCHES:
            p.stop()


class TestTaskStatus:
    def test_pending_value(self):
        assert TaskStatus.PENDING.value == "pending"

    def test_in_progress_value(self):
        assert TaskStatus.IN_PROGRESS.value == "in_progress"

    def test_completed_value(self):
        assert TaskStatus.COMPLETED.value == "completed"

    def test_failed_value(self):
        assert TaskStatus.FAILED.value == "failed"

    def test_skipped_value(self):
        assert TaskStatus.SKIPPED.value == "skipped"

    def test_all_statuses_are_unique(self):
        values = [s.value for s in TaskStatus]
        assert len(values) == len(set(values))

    def test_status_count(self):
        assert len(TaskStatus) == 5

    def test_status_membership(self):
        assert TaskStatus("pending") is TaskStatus.PENDING
        assert TaskStatus("in_progress") is TaskStatus.IN_PROGRESS
        assert TaskStatus("completed") is TaskStatus.COMPLETED
        assert TaskStatus("failed") is TaskStatus.FAILED
        assert TaskStatus("skipped") is TaskStatus.SKIPPED


class TestToolCall:
    def test_basic_creation(self):
        tc = ToolCall(name="test_tool", arguments={"key": "value"})
        assert tc.name == "test_tool"
        assert tc.arguments == {"key": "value"}

    def test_default_fields(self):
        tc = ToolCall(name="tool", arguments={})
        assert tc.id.startswith("call_")
        assert tc.status == TaskStatus.PENDING
        assert tc.result == ""
        assert tc.error == ""
        assert tc.attempts == 0
        assert tc.max_attempts == 3
        assert tc.timestamp != ""

    def test_explicit_id(self):
        tc = ToolCall(name="tool", arguments={}, id="custom_id_123")
        assert tc.id == "custom_id_123"

    def test_auto_generated_id_format(self):
        tc = ToolCall(name="my_tool", arguments={})
        assert tc.id.startswith("call_")
        assert len(tc.id) == 13

    def test_timestamp_format(self):
        tc = ToolCall(name="tool", arguments={})
        parts = tc.timestamp.split(":")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_explicit_timestamp(self):
        tc = ToolCall(name="tool", arguments={}, timestamp="12:34:56")
        assert tc.timestamp == "12:34:56"

    def test_custom_max_attempts(self):
        tc = ToolCall(name="tool", arguments={}, max_attempts=5)
        assert tc.max_attempts == 5

    def test_with_error(self):
        tc = ToolCall(name="tool", arguments={}, error="something broke", status=TaskStatus.FAILED)
        assert tc.error == "something broke"
        assert tc.status == TaskStatus.FAILED

    def test_with_result(self):
        tc = ToolCall(name="tool", arguments={}, result="output data", status=TaskStatus.COMPLETED)
        assert tc.result == "output data"
        assert tc.status == TaskStatus.COMPLETED

    def test_unique_ids(self):
        tc1 = ToolCall(name="tool", arguments={})
        time.sleep(0.01)
        tc2 = ToolCall(name="tool", arguments={})
        assert tc1.id != tc2.id

    def test_empty_arguments(self):
        tc = ToolCall(name="tool", arguments={})
        assert tc.arguments == {}

    def test_nested_arguments(self):
        tc = ToolCall(name="tool", arguments={"nested": {"key": [1, 2, 3]}})
        assert tc.arguments["nested"]["key"] == [1, 2, 3]

    def test_zero_max_attempts(self):
        tc = ToolCall(name="tool", arguments={}, max_attempts=0)
        assert tc.max_attempts == 0

    def test_negative_attempts(self):
        tc = ToolCall(name="tool", arguments={}, attempts=-1)
        assert tc.attempts == -1

    def test_large_max_attempts(self):
        tc = ToolCall(name="tool", arguments={}, max_attempts=100)
        assert tc.max_attempts == 100


class TestPlanStep:
    def test_basic_creation(self):
        ps = PlanStep(description="Step 1: Do something")
        assert ps.description == "Step 1: Do something"

    def test_default_fields(self):
        ps = PlanStep(description="test")
        assert ps.tool_calls == []
        assert ps.status == TaskStatus.PENDING
        assert ps.result == ""
        assert ps.reflection == ""

    def test_with_tool_calls(self):
        tc1 = ToolCall(name="tool1", arguments={"a": 1})
        tc2 = ToolCall(name="tool2", arguments={"b": 2})
        ps = PlanStep(description="Multi-tool step", tool_calls=[tc1, tc2])
        assert len(ps.tool_calls) == 2
        assert ps.tool_calls[0].name == "tool1"
        assert ps.tool_calls[1].name == "tool2"

    def test_with_status(self):
        ps = PlanStep(description="test", status=TaskStatus.COMPLETED)
        assert ps.status == TaskStatus.COMPLETED

    def test_with_reflection(self):
        ps = PlanStep(description="test", reflection="This went well")
        assert ps.reflection == "This went well"

    def test_tool_calls_mutable(self):
        ps = PlanStep(description="test")
        tc = ToolCall(name="tool", arguments={})
        ps.tool_calls.append(tc)
        assert len(ps.tool_calls) == 1


class TestExecutionPlan:
    def test_basic_creation(self):
        ep = ExecutionPlan(goal="Build something great")
        assert ep.goal == "Build something great"

    def test_default_fields(self):
        ep = ExecutionPlan(goal="test")
        assert ep.steps == []
        assert ep.status == TaskStatus.PENDING
        assert ep.final_result == ""
        assert ep.total_tool_calls == 0
        assert ep.successful_calls == 0
        assert ep.failed_calls == 0

    def test_with_steps(self):
        step1 = PlanStep(description="Step 1")
        step2 = PlanStep(description="Step 2")
        ep = ExecutionPlan(goal="Multi-step task", steps=[step1, step2])
        assert len(ep.steps) == 2

    def test_with_stats(self):
        ep = ExecutionPlan(
            goal="test",
            total_tool_calls=10,
            successful_calls=8,
            failed_calls=2,
        )
        assert ep.total_tool_calls == 10
        assert ep.successful_calls == 8
        assert ep.failed_calls == 2

    def test_with_final_result(self):
        ep = ExecutionPlan(goal="test", final_result="All done!")
        assert ep.final_result == "All done!"

    def test_with_status(self):
        ep = ExecutionPlan(goal="test", status=TaskStatus.IN_PROGRESS)
        assert ep.status == TaskStatus.IN_PROGRESS

    def test_steps_mutable(self):
        ep = ExecutionPlan(goal="test")
        step = PlanStep(description="new step")
        ep.steps.append(step)
        assert len(ep.steps) == 1

    def test_many_steps(self):
        steps = [PlanStep(description=f"step_{i}") for i in range(100)]
        ep = ExecutionPlan(goal="stress test", steps=steps)
        assert len(ep.steps) == 100

    def test_negative_stats(self):
        ep = ExecutionPlan(goal="test", total_tool_calls=-1, successful_calls=-2, failed_calls=-3)
        assert ep.total_tool_calls == -1
        assert ep.successful_calls == -2
        assert ep.failed_calls == -3

    def test_long_goal(self):
        long_goal = "A" * 10000
        ep = ExecutionPlan(goal=long_goal)
        assert len(ep.goal) == 10000


class TestAgentClassConstants:
    def test_max_tool_loops(self):
        assert Agent.MAX_TOOL_LOOPS == 10

    def test_max_retries(self):
        assert Agent.MAX_RETRIES == 3

    def test_retry_backoff_base(self):
        assert Agent.RETRY_BACKOFF_BASE == 2

    def test_context_reserved_tokens(self):
        assert Agent.CONTEXT_RESERVED_TOKENS == 1500


class TestAgentInitialization:
    def test_agent_creates_with_defaults(self):
        agent = make_agent()
        assert agent.model is not None
        assert agent.memory is not None
        assert agent.tools is not None
        assert agent.context is not None

    def test_agent_has_plugins_attribute(self):
        agent = make_agent()
        assert hasattr(agent, "plugins")

    def test_agent_has_execution_history(self):
        agent = make_agent()
        assert hasattr(agent, "_execution_history")
        assert agent._execution_history == []

    def test_agent_has_retriever_attribute(self):
        agent = make_agent()
        assert hasattr(agent, "_retriever")

    def test_agent_has_fast_mode_attribute(self):
        agent = make_agent()
        assert hasattr(agent, "_fast_mode")

    def test_agent_has_tools_attribute(self):
        agent = make_agent()
        assert hasattr(agent, "tools")

    def test_agent_has_memory_attribute(self):
        agent = make_agent()
        assert hasattr(agent, "memory")

    def test_agent_has_context_attribute(self):
        agent = make_agent()
        assert hasattr(agent, "context")

    def test_agent_has_cache_attribute(self):
        agent = make_agent()
        assert hasattr(agent, "_cache")

    def test_agent_accepts_custom_model(self):
        patches = [p.start() for p in AGENT_PATCHES]
        mock_cache = patches[3]
        mock_cache.return_value.get_cache.return_value = MagicMock()
        try:
            custom_model = MagicMock()
            agent = Agent(model=custom_model)
            assert agent.model is custom_model
        finally:
            for p in AGENT_PATCHES:
                p.stop()

    def test_agent_accepts_custom_memory(self):
        patches = [p.start() for p in AGENT_PATCHES]
        mock_cache = patches[3]
        mock_cache.return_value.get_cache.return_value = MagicMock()
        try:
            custom_memory = MagicMock()
            agent = Agent(memory=custom_memory)
            assert agent.memory is custom_memory
        finally:
            for p in AGENT_PATCHES:
                p.stop()

    def test_agent_accepts_custom_tools(self):
        patches = [p.start() for p in AGENT_PATCHES]
        mock_cache = patches[3]
        mock_cache.return_value.get_cache.return_value = MagicMock()
        try:
            custom_tools = MagicMock()
            agent = Agent(tools=custom_tools)
            assert agent.tools is custom_tools
        finally:
            for p in AGENT_PATCHES:
                p.stop()

    def test_agent_accepts_custom_context(self):
        patches = [p.start() for p in AGENT_PATCHES]
        mock_cache = patches[3]
        mock_cache.return_value.get_cache.return_value = MagicMock()
        try:
            custom_context = MagicMock()
            agent = Agent(context=custom_context)
            assert agent.context is custom_context
        finally:
            for p in AGENT_PATCHES:
                p.stop()


class TestAgentIsSimpleQuery:
    def test_simple_query_returns_true(self):
        agent = make_agent()
        assert agent._is_simple_query("hello there") is True

    def test_tool_keyword_returns_false(self):
        agent = make_agent()
        assert agent._is_simple_query("read the file") is False

    def test_long_query_returns_false(self):
        agent = make_agent()
        long_query = " ".join(["word"] * 15)
        assert agent._is_simple_query(long_query) is False

    def test_search_keyword_returns_false(self):
        agent = make_agent()
        assert agent._is_simple_query("search for files") is False

    def test_git_keyword_returns_false(self):
        agent = make_agent()
        assert agent._is_simple_query("git commit") is False


class TestAgentToolParsing:
    def test_parse_json_tool_calls(self):
        agent = make_agent()
        text = '{"tool_calls": [{"name": "read_file", "arguments": {"path": "/tmp/test"}}]}'
        calls = agent._parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "read_file"
        assert calls[0].arguments == {"path": "/tmp/test"}

    def test_parse_code_fence_json(self):
        agent = make_agent()
        text = '```json\n{"tool_calls": [{"name": "write_file", "arguments": {"content": "hello"}}]}\n```'
        calls = agent._parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "write_file"

    def test_parse_legacy_tool_format(self):
        agent = make_agent()
        text = '<tool name="search">query=hello</tool>'
        calls = agent._parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0].name == "search"
        assert calls[0].arguments["query"] == "hello"

    def test_parse_no_tool_calls(self):
        agent = make_agent()
        text = "This is just a regular response with no tool calls."
        calls = agent._parse_tool_calls(text)
        assert calls == []


class TestAgentGetExecutionHistory:
    def test_empty_history(self):
        agent = make_agent()
        history = agent.get_execution_history()
        assert history == []

    def test_history_after_plan_execution(self):
        agent = make_agent()
        plan = ExecutionPlan(
            goal="test goal",
            status=TaskStatus.COMPLETED,
            total_tool_calls=3,
            successful_calls=2,
            failed_calls=1,
            steps=[PlanStep(description="step1"), PlanStep(description="step2")],
        )
        agent._execution_history.append(plan)
        history = agent.get_execution_history()
        assert len(history) == 1
        assert history[0]["goal"] == "test goal"
        assert history[0]["status"] == "completed"
        assert history[0]["total_calls"] == 3
        assert history[0]["successful"] == 2
        assert history[0]["failed"] == 1
        assert history[0]["steps"] == 2


class TestAgentGetCacheStats:
    def test_cache_stats_returns_dict(self):
        patches = [p.start() for p in AGENT_PATCHES]
        mock_cache = patches[3]
        mock_cache.return_value.get_cache.return_value = MagicMock()
        mock_cache.return_value.get_cache.return_value.get_stats.return_value = {"hits": 5, "misses": 2}
        try:
            agent = Agent()
            stats = agent.get_cache_stats()
            assert isinstance(stats, dict)
            assert stats["hits"] == 5
        finally:
            for p in AGENT_PATCHES:
                p.stop()
