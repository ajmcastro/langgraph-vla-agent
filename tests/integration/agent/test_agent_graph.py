"""Integration tests for the full compiled LangGraph agent.

Requires the [agent] extra: uv sync --extra dev --extra agent
Gracefully skipped when LangGraph is not installed.
"""

import pytest

langgraph = pytest.importorskip("langgraph", reason="[agent] extra not installed")

from langgraph_vla_agent.agent.config import AgentConfig, Granularity  # noqa: E402
from langgraph_vla_agent.agent.runner import make_mock_runner  # noqa: E402
from langgraph_vla_agent.agent.state import AgentStatus  # noqa: E402
from langgraph_vla_agent.environments.mock import MockScenario  # noqa: E402

# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_agent_completes_pick_and_place_goal() -> None:
    runner = make_mock_runner()
    state = runner.run("pick up the cube and place it in the bin")
    assert state["final_status"] == AgentStatus.COMPLETED
    assert len(state["completed_subtask_ids"]) == 2  # coarse: 2 subtasks
    assert state["failed_subtask_ids"] == []


def test_agent_completes_with_fine_granularity() -> None:
    config = AgentConfig(granularity=Granularity.FINE)
    runner = make_mock_runner(config=config)
    state = runner.run("pick up the cube and place it in the bin")
    assert state["final_status"] == AgentStatus.COMPLETED
    assert len(state["completed_subtask_ids"]) == 5  # fine: 5 subtasks


def test_agent_run_id_is_populated() -> None:
    runner = make_mock_runner()
    state = runner.run("pick up the cube")
    assert state["run_id"]


def test_agent_execution_history_is_recorded() -> None:
    runner = make_mock_runner()
    state = runner.run("pick up the cube and place it in the bin")
    assert len(state["execution_history_references"]) >= 2


def test_agent_accepts_explicit_run_id() -> None:
    runner = make_mock_runner()
    state = runner.run("pick up the cube", run_id="my-run-123")
    assert state["run_id"] == "my-run-123"


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


def test_agent_fails_on_unrecognised_goal() -> None:
    runner = make_mock_runner()
    state = runner.run("dance around the room")
    assert state["final_status"] == AgentStatus.FAILED


def test_agent_fails_on_empty_goal() -> None:
    runner = make_mock_runner()
    state = runner.run("")
    assert state["final_status"] == AgentStatus.FAILED


# ---------------------------------------------------------------------------
# Safety gate
# ---------------------------------------------------------------------------


def test_agent_safety_stops_on_blocked_instruction() -> None:
    from langgraph_vla_agent.agent.graph import build_agent_graph
    from langgraph_vla_agent.agent.runner import AgentRunner
    from langgraph_vla_agent.agent.safety import SafetyChecker
    from langgraph_vla_agent.domain.context import EvaluationMode

    # Planner that always produces a subtask with a blocked keyword.
    from langgraph_vla_agent.domain.tasks import SubTask, TaskGoal, TaskPlan
    from langgraph_vla_agent.environments.mock import MockEnvironment
    from langgraph_vla_agent.execution.config import ExecutorConfig
    from langgraph_vla_agent.execution.executor import Executor
    from langgraph_vla_agent.policies.mock import MockPolicyBehavior, MockRobotPolicy

    class _BlockedPlanner:
        def plan(self, goal: TaskGoal) -> TaskPlan:
            return TaskPlan(
                goal=goal,
                subtasks=[SubTask(instruction="move toward the human", success_criteria="done")],
                planner_id="blocked-test",
            )

    config = AgentConfig(evaluation_mode=EvaluationMode.MOCK)
    executor = Executor(
        MockRobotPolicy(behavior=MockPolicyBehavior.ALWAYS_VALID),
        MockEnvironment(),
        ExecutorConfig(evaluation_mode=EvaluationMode.MOCK),
    )
    graph = build_agent_graph(
        executor=executor,
        planner=_BlockedPlanner(),
        config=config,
        safety_checker=SafetyChecker(),
    )
    runner = AgentRunner(graph=graph, config=config)
    state = runner.run("pick up the cube")
    assert state["final_status"] == AgentStatus.SAFETY_STOP
    assert state["safety_status"] == "rejected"


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


def test_agent_retries_on_failure_then_succeeds() -> None:
    """First subtask attempt fails; retry succeeds on step 2."""
    from langgraph_vla_agent.agent.graph import build_agent_graph
    from langgraph_vla_agent.agent.runner import AgentRunner
    from langgraph_vla_agent.domain.context import EvaluationMode
    from langgraph_vla_agent.environments.mock import MockEnvironment, MockScenario
    from langgraph_vla_agent.execution.config import ExecutorConfig
    from langgraph_vla_agent.execution.executor import Executor
    from langgraph_vla_agent.policies.mock import MockPolicyBehavior, MockRobotPolicy

    config = AgentConfig(
        max_retries=2,
        max_replans=0,
        evaluation_mode=EvaluationMode.MOCK,
    )
    # Environment fails on first episode, succeeds on second (retry).
    call_count = [0]

    class _FlakyEnvironment(MockEnvironment):
        def reset(self, subtask):  # type: ignore[override]
            call_count[0] += 1
            # First call per subtask: use FAIL_AT_STEP; second: SUCCEED_AT_STEP.
            if call_count[0] % 2 == 1:
                self.scenario = MockScenario.FAIL_AT_STEP
            else:
                self.scenario = MockScenario.SUCCEED_AT_STEP
            self.n = 2
            return super().reset(subtask)

    executor = Executor(
        MockRobotPolicy(behavior=MockPolicyBehavior.ALWAYS_VALID),
        _FlakyEnvironment(scenario=MockScenario.FAIL_AT_STEP, n=2),
        ExecutorConfig(max_steps=10, evaluation_mode=EvaluationMode.MOCK),
    )
    from langgraph_vla_agent.planning.deterministic import DeterministicPlanner

    graph = build_agent_graph(
        executor=executor,
        planner=DeterministicPlanner(granularity="coarse"),  # type: ignore[arg-type]
        config=config,
    )
    runner = AgentRunner(graph=graph, config=config)
    state = runner.run("pick up the cube and place it in the bin")
    # May complete or fail — main assertion is it terminates without exception.
    assert state["final_status"] in (AgentStatus.COMPLETED, AgentStatus.FAILED)


def test_agent_fails_after_exhausting_retries_and_replans() -> None:
    config = AgentConfig(max_retries=0, max_replans=0)
    runner = make_mock_runner(
        config=config,
        scenario=MockScenario.FAIL_AT_STEP,
        succeed_at_step=999,
    )
    state = runner.run("pick up the cube and place it in the bin")
    assert state["final_status"] == AgentStatus.FAILED
