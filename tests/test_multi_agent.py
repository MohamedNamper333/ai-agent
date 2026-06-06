"""Tests for tools.multi_agent (Task 14)."""
import json
import re
import unittest
from unittest.mock import MagicMock, patch

from tools.multi_agent import MultiAgentOrchestrator, SpecialistAgent


def _make_mock_model(response=""):
    """Create a mock LLM model with given response."""
    mock = MagicMock()
    mock.generate.return_value = response
    return mock


class TestSpecialistAgentInit(unittest.TestCase):
    def test_stores_all_attributes(self):
        mock = _make_mock_model()
        agent = SpecialistAgent(
            "The Analyst",
            "Data analyst",
            "Expert in data analysis",
            "Be thorough",
            model=mock,
        )
        self.assertEqual(agent.name, "The Analyst")
        self.assertEqual(agent.role, "Data analyst")
        self.assertEqual(agent.expertise, "Expert in data analysis")
        self.assertEqual(agent.instructions, "Be thorough")
        self.assertIs(agent.model, mock)

    def test_creates_default_llm_when_no_model(self):
        with patch("tools.multi_agent.LLM") as MockLLM:
            SpecialistAgent("X", "R", "E", "I")
            MockLLM.assert_called_once()

    def test_memory_starts_empty(self):
        agent = SpecialistAgent("X", "R", "E", "I", model=_make_mock_model())
        self.assertEqual(agent.get_history(), [])


class TestSpecialistAgentProcess(unittest.TestCase):
    def setUp(self):
        self.mock = _make_mock_model("Agent response")
        self.agent = SpecialistAgent(
            "The Analyst",
            "Data analyst",
            "Expert in data",
            "Be thorough",
            model=self.mock,
        )

    def test_generate_called_with_max_tokens_2000(self):
        self.agent.process("test task")
        self.mock.generate.assert_called_once()
        _, kwargs = self.mock.generate.call_args
        self.assertEqual(kwargs.get("max_tokens"), 2000)

    def test_prompt_contains_name(self):
        self.agent.process("test task")
        prompt = self.mock.generate.call_args[0][0]
        self.assertIn("The Analyst", prompt)

    def test_prompt_contains_role(self):
        self.agent.process("test task")
        prompt = self.mock.generate.call_args[0][0]
        self.assertIn("Data analyst", prompt)

    def test_prompt_contains_expertise(self):
        self.agent.process("test task")
        prompt = self.mock.generate.call_args[0][0]
        self.assertIn("Expert in data", prompt)

    def test_prompt_contains_instructions(self):
        self.agent.process("test task")
        prompt = self.mock.generate.call_args[0][0]
        self.assertIn("Be thorough", prompt)

    def test_prompt_contains_task(self):
        self.agent.process("my specific task")
        prompt = self.mock.generate.call_args[0][0]
        self.assertIn("my specific task", prompt)

    def test_context_included_when_provided(self):
        self.agent.process("task", context="some context here")
        prompt = self.mock.generate.call_args[0][0]
        self.assertIn("some context here", prompt)

    def test_context_omitted_when_empty(self):
        self.agent.process("task", context="")
        prompt = self.mock.generate.call_args[0][0]
        self.assertNotIn("Context:", prompt)

    def test_tools_included_when_provided(self):
        self.agent.process("task", tools="search tool")
        prompt = self.mock.generate.call_args[0][0]
        self.assertIn("search tool", prompt)

    def test_tools_omitted_when_empty(self):
        self.agent.process("task", tools="")
        prompt = self.mock.generate.call_args[0][0]
        self.assertNotIn("Tools:", prompt)

    def test_previous_analyses_included_when_provided(self):
        self.agent.process("task", previous_analyses="prior work")
        prompt = self.mock.generate.call_args[0][0]
        self.assertIn("prior work", prompt)

    def test_previous_analyses_omitted_when_empty(self):
        self.agent.process("task", previous_analyses="")
        prompt = self.mock.generate.call_args[0][0]
        self.assertNotIn("Previous Analyses:", prompt)

    def test_memory_appends_after_process(self):
        self.agent.process("test task")
        history = self.agent.get_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["task"], "test task")
        self.assertEqual(history[0]["response"], "Agent response")

    def test_memory_truncates_long_task(self):
        long_task = "x" * 200
        self.agent.process(long_task)
        history = self.agent.get_history()
        self.assertEqual(len(history[0]["task"]), 100)

    def test_memory_truncates_long_response(self):
        long_response = "y" * 600
        self.mock.generate.return_value = long_response
        self.agent.process("task")
        history = self.agent.get_history()
        self.assertEqual(len(history[0]["response"]), 500)

    def test_memory_has_timestamp(self):
        self.agent.process("task")
        history = self.agent.get_history()
        self.assertIn("timestamp", history[0])
        self.assertRegex(history[0]["timestamp"], r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}")

    def test_returns_response(self):
        self.mock.generate.return_value = "my answer"
        self.assertEqual(self.agent.process("task"), "my answer")


class TestSpecialistAgentGetHistory(unittest.TestCase):
    def test_empty_initially(self):
        agent = SpecialistAgent("X", "R", "E", "I", model=_make_mock_model())
        self.assertEqual(agent.get_history(), [])

    def test_returns_copy(self):
        agent = SpecialistAgent("X", "R", "E", "I", model=_make_mock_model())
        agent.process("task")
        history = agent.get_history()
        history.append({"fake": "entry"})
        self.assertEqual(len(agent.get_history()), 1)

    def test_reflects_process_calls(self):
        agent = SpecialistAgent("X", "R", "E", "I", model=_make_mock_model())
        agent.process("task1")
        agent.process("task2")
        history = agent.get_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["task"], "task1")
        self.assertEqual(history[1]["task"], "task2")


class TestMultiAgentOrchestratorInit(unittest.TestCase):
    def test_uses_explicit_model(self):
        mock = _make_mock_model()
        orch = MultiAgentOrchestrator(model=mock)
        self.assertIs(orch.model, mock)

    def test_creates_default_llm(self):
        with patch("tools.multi_agent.LLM") as MockLLM:
            MultiAgentOrchestrator()
            MockLLM.assert_called_once()

    def test_creates_four_specialists(self):
        orch = MultiAgentOrchestrator(model=_make_mock_model())
        self.assertEqual(len(orch.specialists), 4)
        for s in orch.specialists:
            self.assertIsInstance(s, SpecialistAgent)

    def test_specialist_names(self):
        orch = MultiAgentOrchestrator(model=_make_mock_model())
        names = [s.name for s in orch.specialists]
        self.assertIn("The Analyst", names)
        self.assertIn("The Programmer", names)
        self.assertIn("The Reviewer", names)
        self.assertIn("The Architect", names)

    def test_all_specialists_share_model(self):
        mock = _make_mock_model()
        orch = MultiAgentOrchestrator(model=mock)
        for s in orch.specialists:
            self.assertIs(s.model, mock)


class TestMultiAgentOrchestratorDelegate(unittest.TestCase):
    def setUp(self):
        self.orch = MultiAgentOrchestrator(model=_make_mock_model("ok"))

    def test_known_agent_runs(self):
        self.assertEqual(self.orch.delegate("The Analyst", "task"), "ok")

    def test_case_insensitive_match(self):
        self.assertEqual(self.orch.delegate("the analyst", "task"), "ok")

    def test_unknown_agent_returns_error(self):
        result = self.orch.delegate("The Unknown", "task")
        self.assertIn("Unknown specialist", result)

    def test_error_message_lists_available(self):
        result = self.orch.delegate("The Unknown", "task")
        self.assertIn("The Analyst", result)
        self.assertIn("The Programmer", result)
        self.assertIn("The Reviewer", result)
        self.assertIn("The Architect", result)

    def test_passes_context_to_specialist(self):
        self.orch.delegate("The Analyst", "task", context="my context")
        analyst = next(s for s in self.orch.specialists if s.name == "The Analyst")
        prompt = analyst.model.generate.call_args[0][0]
        self.assertIn("my context", prompt)

    def test_passes_tools_to_specialist(self):
        self.orch.delegate("The Analyst", "task", tools="search")
        analyst = next(s for s in self.orch.specialists if s.name == "The Analyst")
        prompt = analyst.model.generate.call_args[0][0]
        self.assertIn("search", prompt)


class TestMultiAgentOrchestratorRunCouncil(unittest.TestCase):
    def setUp(self):
        self.orch = MultiAgentOrchestrator(model=_make_mock_model("synthesis"))

    def test_returns_synthesis(self):
        self.assertEqual(self.orch.run_council("my task"), "synthesis")

    def test_parallel_default_calls_5_times(self):
        self.orch.run_council("task")
        self.assertEqual(self.orch.model.generate.call_count, 5)

    def test_sequential_calls_5_times(self):
        self.orch.run_council("task", parallel=False)
        self.assertEqual(self.orch.model.generate.call_count, 5)

    def test_synthesis_prompt_includes_all_names(self):
        self.orch.run_council("task")
        synth_prompt = self.orch.model.generate.call_args_list[-1][0][0]
        for name in ["The Analyst", "The Programmer", "The Reviewer", "The Architect"]:
            self.assertIn(name, synth_prompt)

    def test_synthesis_uses_max_tokens_2500(self):
        self.orch.run_council("task")
        synth_kwargs = self.orch.model.generate.call_args_list[-1][1]
        self.assertEqual(synth_kwargs.get("max_tokens"), 2500)


class TestMultiAgentOrchestratorSequential(unittest.TestCase):
    def setUp(self):
        self.orch = MultiAgentOrchestrator(
            model=_make_mock_model("sequential answer")
        )

    def test_five_total_calls(self):
        self.orch.run_council("task", parallel=False)
        self.assertEqual(self.orch.model.generate.call_count, 5)

    def test_previous_accumulator_passed(self):
        responses = ["resp_A", "resp_B", "resp_C", "resp_D", "synth"]
        self.orch.model.generate.side_effect = responses
        self.orch.run_council("task", parallel=False)
        second_prompt = self.orch.model.generate.call_args_list[1][0][0]
        self.assertIn("resp_A", second_prompt)
        self.assertIn("The Analyst", second_prompt)

    def test_truncates_previous_at_300(self):
        long_response = "z" * 400
        self.orch.model.generate.side_effect = [
            long_response, "second", "third", "fourth", "synth"
        ]
        self.orch.run_council("task", parallel=False)
        second_prompt = self.orch.model.generate.call_args_list[1][0][0]
        self.assertIn("z" * 300, second_prompt)
        self.assertNotIn("z" * 301, second_prompt)


class TestMultiAgentOrchestratorParallel(unittest.TestCase):
    def setUp(self):
        self.orch = MultiAgentOrchestrator(
            model=_make_mock_model("parallel answer")
        )

    def test_uses_thread_pool_executor(self):
        with patch("tools.multi_agent.ThreadPoolExecutor") as MockExecutor, \
             patch("tools.multi_agent.as_completed", return_value=[]):
            mock_exec_instance = MagicMock()
            mock_exec_instance.__enter__ = MagicMock(
                return_value=mock_exec_instance
            )
            mock_exec_instance.__exit__ = MagicMock(return_value=False)
            MockExecutor.return_value = mock_exec_instance
            self.orch.run_council("task", parallel=True)
            MockExecutor.assert_called_once()
            call_kwargs = MockExecutor.call_args[1]
            self.assertEqual(call_kwargs.get("max_workers"), 4)

    def test_exception_caught_per_agent(self):
        def raise_exc(*args, **kwargs):
            raise RuntimeError("boom")
        self.orch.specialists[0].process = raise_exc
        result = self.orch.run_council("task", parallel=True)
        self.assertIsNotNone(result)

    def test_synthesis_still_called_on_exception(self):
        def raise_exc(*args, **kwargs):
            raise RuntimeError("boom")
        self.orch.specialists[0].process = raise_exc
        self.orch.run_council("task", parallel=True)
        self.assertGreaterEqual(self.orch.model.generate.call_count, 1)


class TestMultiAgentOrchestratorDebate(unittest.TestCase):
    def setUp(self):
        self.orch = MultiAgentOrchestrator(
            model=_make_mock_model("debate summary")
        )

    def test_default_2_rounds_calls_9_times(self):
        self.orch.debate("topic")
        self.assertEqual(self.orch.model.generate.call_count, 9)

    def test_custom_rounds(self):
        self.orch.debate("topic", rounds=3)
        self.assertEqual(self.orch.model.generate.call_count, 13)

    def test_returns_model_output(self):
        self.assertEqual(self.orch.debate("topic"), "debate summary")

    def test_subsequent_agents_see_others_responses(self):
        self.orch.debate("topic", rounds=1)
        second_prompt = self.orch.model.generate.call_args_list[1][0][0]
        self.assertIn("debate summary", second_prompt)


class TestMultiAgentOrchestratorGroupConsensus(unittest.TestCase):
    def setUp(self):
        self.orch = MultiAgentOrchestrator(
            model=_make_mock_model(
                json.dumps({"vote": 1, "reason": "good", "confidence": 0.8})
            )
        )

    def test_no_consensus_on_no_json(self):
        self.orch.model.generate.return_value = "no json here"
        result = self.orch.group_consensus("task", ["A", "B"])
        self.assertIn("No consensus", result)

    def test_parses_valid_json(self):
        result = self.orch.group_consensus("task", ["A", "B"])
        self.assertIn("A", result)

    def test_returns_winner_text(self):
        result = self.orch.group_consensus("task", ["A", "B"])
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_weights_by_confidence(self):
        responses = [
            json.dumps({"vote": 1, "reason": "r1", "confidence": 0.3}),
            json.dumps({"vote": 2, "reason": "r2", "confidence": 0.9}),
            json.dumps({"vote": 1, "reason": "r3", "confidence": 0.4}),
            json.dumps({"vote": 2, "reason": "r4", "confidence": 0.8}),
        ]
        self.orch.model.generate.side_effect = responses
        result = self.orch.group_consensus("task", ["A", "B"])
        self.assertIn("B", result)


class TestMultiAgentOrchestratorGetSpecialistInfo(unittest.TestCase):
    def setUp(self):
        self.orch = MultiAgentOrchestrator(model=_make_mock_model("ok"))

    def test_returns_four_entries(self):
        self.assertEqual(len(self.orch.get_specialist_info()), 4)

    def test_required_keys(self):
        for entry in self.orch.get_specialist_info():
            self.assertIn("name", entry)
            self.assertIn("role", entry)
            self.assertIn("expertise", entry)
            self.assertIn("tasks_completed", entry)

    def test_tasks_completed_starts_at_zero(self):
        for entry in self.orch.get_specialist_info():
            self.assertEqual(entry["tasks_completed"], 0)

    def test_expertise_truncated_at_80(self):
        for entry in self.orch.get_specialist_info():
            self.assertLessEqual(len(entry["expertise"]), 80)

    def test_updates_after_process(self):
        self.orch.delegate("The Analyst", "task")
        info = self.orch.get_specialist_info()
        analyst_info = next(e for e in info if e["name"] == "The Analyst")
        self.assertEqual(analyst_info["tasks_completed"], 1)


if __name__ == "__main__":
    unittest.main()
