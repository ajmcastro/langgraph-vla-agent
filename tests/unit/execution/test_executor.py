"""Unit tests for the Executor loop.

Each test exercises one terminal path through the executor. All paths are
reachable without GPU, network, or dataset — only MockRobotPolicy and
MockEnvironment are used.
"""

from langgraph_vla_agent.domain import EvaluationMode, PolicyContext, SubTask
from langgraph_vla_agent.domain.results import ExecutionStatus, FailureReason
from langgraph_vla_agent.environments import MockEnvironment, MockScenario
from langgraph_vla_agent.execution import Executor, ExecutorConfig
from langgraph_vla_agent.policies import MockPolicyBehavior, MockRobotPolicy


def _context(episode_id: str = "ep-1") -> PolicyContext:
    return PolicyContext(
        run_id="run-test",
        episode_id=episode_id,
        evaluation_mode=EvaluationMode.MOCK,
        seed=0,
    )


def _subtask(instruction: str = "grasp the cube") -> SubTask:
    return SubTask(id="st-fixed", instruction=instruction)


def _config(**kwargs: object) -> ExecutorConfig:
    return ExecutorConfig(evaluation_mode=EvaluationMode.MOCK, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# SUCCESS path
# ---------------------------------------------------------------------------


def test_executor_success_path() -> None:
    policy = MockRobotPolicy()
    env = MockEnvironment(scenario=MockScenario.SUCCEED_AT_STEP, n=3)
    executor = Executor(policy, env, _config(max_steps=20))

    result = executor.run(_subtask(), _context())

    assert result.status == ExecutionStatus.SUCCESS
    assert result.failure_reason == FailureReason.NONE
    assert result.steps_taken == 3
    assert result.subtask_id == "st-fixed"
    assert result.evaluation_mode == EvaluationMode.MOCK
    assert result.succeeded


def test_executor_result_contains_elapsed_metric() -> None:
    policy = MockRobotPolicy()
    env = MockEnvironment(scenario=MockScenario.SUCCEED_AT_STEP, n=1)
    result = Executor(policy, env, _config()).run(_subtask(), _context())
    assert "elapsed_s" in result.metrics
    assert result.metrics["elapsed_s"] >= 0.0


# ---------------------------------------------------------------------------
# FAILURE path (environment signals failure)
# ---------------------------------------------------------------------------


def test_executor_failure_path() -> None:
    policy = MockRobotPolicy()
    env = MockEnvironment(scenario=MockScenario.FAIL_AT_STEP, n=2)
    result = Executor(policy, env, _config(max_steps=10)).run(_subtask(), _context())

    assert result.status == ExecutionStatus.FAILURE
    assert result.failure_reason == FailureReason.UNKNOWN
    assert result.steps_taken == 2
    assert not result.succeeded


# ---------------------------------------------------------------------------
# MAX_STEPS path — executor loop exhausts its budget
# ---------------------------------------------------------------------------


def test_executor_max_steps_exceeded_via_config() -> None:
    policy = MockRobotPolicy()
    env = MockEnvironment(scenario=MockScenario.NEVER_TERMINATE)
    result = Executor(policy, env, _config(max_steps=5)).run(_subtask(), _context())

    assert result.status == ExecutionStatus.MAX_STEPS_EXCEEDED
    assert result.failure_reason == FailureReason.MAX_STEPS_EXCEEDED
    assert result.steps_taken == 5


# ---------------------------------------------------------------------------
# MAX_STEPS path — environment signals truncation
# ---------------------------------------------------------------------------


def test_executor_truncation_path() -> None:
    policy = MockRobotPolicy()
    env = MockEnvironment(scenario=MockScenario.TRUNCATE_AT_STEP, n=4)
    result = Executor(policy, env, _config(max_steps=20)).run(_subtask(), _context())

    assert result.status == ExecutionStatus.MAX_STEPS_EXCEEDED
    assert result.steps_taken == 4


# ---------------------------------------------------------------------------
# INVALID_ACTION path — policy returns NaN
# ---------------------------------------------------------------------------


def test_executor_invalid_action_path() -> None:
    policy = MockRobotPolicy(behavior=MockPolicyBehavior.INVALID_AFTER_N, n=2)
    env = MockEnvironment(scenario=MockScenario.NEVER_TERMINATE)
    result = Executor(policy, env, _config(max_steps=10)).run(_subtask(), _context())

    assert result.status == ExecutionStatus.INVALID_ACTION
    assert result.failure_reason == FailureReason.INVALID_ACTION
    assert result.steps_taken == 3  # steps 1-2 valid, step 3 triggers NaN


def test_executor_validation_disabled_passes_nan() -> None:
    """With validate_actions=False, NaN actions reach the environment."""
    policy = MockRobotPolicy(behavior=MockPolicyBehavior.INVALID_AFTER_N, n=1)
    env = MockEnvironment(scenario=MockScenario.SUCCEED_AT_STEP, n=3)
    result = Executor(policy, env, _config(max_steps=10, validate_actions=False)).run(
        _subtask(), _context()
    )
    # Env doesn't care about NaN — it still succeeds at step 3
    assert result.status == ExecutionStatus.SUCCESS


# ---------------------------------------------------------------------------
# POLICY_ERROR path — policy raises an exception
# ---------------------------------------------------------------------------


def test_executor_policy_error_path() -> None:
    policy = MockRobotPolicy(behavior=MockPolicyBehavior.RAISE_AFTER_N, n=1)
    env = MockEnvironment(scenario=MockScenario.NEVER_TERMINATE)
    result = Executor(policy, env, _config(max_steps=10)).run(_subtask(), _context())

    assert result.status == ExecutionStatus.FAILURE
    assert result.failure_reason == FailureReason.POLICY_ERROR
    assert result.steps_taken == 1  # failed before step 2 completed


# ---------------------------------------------------------------------------
# Executor calls policy.reset() with the correct context
# ---------------------------------------------------------------------------


def test_executor_passes_context_to_policy_reset() -> None:
    policy = MockRobotPolicy()
    env = MockEnvironment(scenario=MockScenario.SUCCEED_AT_STEP, n=1)
    ctx = _context(episode_id="ep-special")
    Executor(policy, env, _config()).run(_subtask(), ctx)
    assert policy.last_context is ctx


# ---------------------------------------------------------------------------
# Executor propagates evaluation_mode into result
# ---------------------------------------------------------------------------


def test_executor_evaluation_mode_in_result() -> None:
    policy = MockRobotPolicy()
    env = MockEnvironment(scenario=MockScenario.SUCCEED_AT_STEP, n=1)
    config = ExecutorConfig(evaluation_mode=EvaluationMode.REPLAY, max_steps=5)
    result = Executor(policy, env, config).run(_subtask(), _context())
    assert result.evaluation_mode == EvaluationMode.REPLAY


# ---------------------------------------------------------------------------
# Multiple independent runs don't share state
# ---------------------------------------------------------------------------


def test_executor_runs_are_independent() -> None:
    policy = MockRobotPolicy()
    env = MockEnvironment(scenario=MockScenario.SUCCEED_AT_STEP, n=2)
    executor = Executor(policy, env, _config())

    r1 = executor.run(_subtask("first task"), _context("ep-1"))
    r2 = executor.run(_subtask("second task"), _context("ep-2"))

    assert r1.status == ExecutionStatus.SUCCESS
    assert r2.status == ExecutionStatus.SUCCESS
    assert r1.steps_taken == r2.steps_taken == 2
