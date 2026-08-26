"""Unit tests for MockEnvironment."""

import numpy as np

from langgraph_vla_agent.domain import RobotAction, SubTask
from langgraph_vla_agent.environments import MockEnvironment, MockScenario, RobotEnvironment


def _make_subtask(instruction: str = "test task") -> SubTask:
    return SubTask(instruction=instruction)


def _make_action() -> RobotAction:
    return RobotAction(values=np.zeros(6, dtype=np.float32))


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_mock_environment_satisfies_protocol() -> None:
    env = MockEnvironment()
    assert isinstance(env, RobotEnvironment)


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


def test_reset_returns_valid_observation() -> None:
    env = MockEnvironment(state_dim=6)
    obs = env.reset(_make_subtask())
    assert obs.state.shape == (6,)
    assert obs.timestamp == 0.0


def test_reset_clears_step_count() -> None:
    env = MockEnvironment(scenario=MockScenario.NEVER_TERMINATE)
    env.reset(_make_subtask())
    for _ in range(5):
        env.step(_make_action())
    assert env.step_count == 5

    env.reset(_make_subtask())
    assert env.step_count == 0


# ---------------------------------------------------------------------------
# SUCCEED_AT_STEP scenario
# ---------------------------------------------------------------------------


def test_succeed_at_step_not_terminal_before_n() -> None:
    env = MockEnvironment(scenario=MockScenario.SUCCEED_AT_STEP, n=3)
    env.reset(_make_subtask())
    for _ in range(2):
        _, result = env.step(_make_action())
        assert not result.terminated
        assert not result.success


def test_succeed_at_step_terminates_at_n() -> None:
    env = MockEnvironment(scenario=MockScenario.SUCCEED_AT_STEP, n=3)
    env.reset(_make_subtask())
    for _ in range(2):
        env.step(_make_action())
    _, result = env.step(_make_action())  # step 3
    assert result.terminated
    assert result.success
    assert not result.truncated


# ---------------------------------------------------------------------------
# FAIL_AT_STEP scenario
# ---------------------------------------------------------------------------


def test_fail_at_step_terminates_without_success() -> None:
    env = MockEnvironment(scenario=MockScenario.FAIL_AT_STEP, n=2)
    env.reset(_make_subtask())
    env.step(_make_action())  # step 1
    _, result = env.step(_make_action())  # step 2
    assert result.terminated
    assert not result.success


# ---------------------------------------------------------------------------
# TRUNCATE_AT_STEP scenario
# ---------------------------------------------------------------------------


def test_truncate_at_step_emits_truncated_not_terminated() -> None:
    env = MockEnvironment(scenario=MockScenario.TRUNCATE_AT_STEP, n=2)
    env.reset(_make_subtask())
    env.step(_make_action())
    _, result = env.step(_make_action())  # step 2
    assert result.truncated
    assert not result.terminated
    assert not result.success


# ---------------------------------------------------------------------------
# NEVER_TERMINATE scenario
# ---------------------------------------------------------------------------


def test_never_terminate_does_not_terminate() -> None:
    env = MockEnvironment(scenario=MockScenario.NEVER_TERMINATE, n=1)
    env.reset(_make_subtask())
    for _ in range(20):
        _, result = env.step(_make_action())
        assert not result.terminated
        assert not result.truncated


# ---------------------------------------------------------------------------
# observe()
# ---------------------------------------------------------------------------


def test_observe_does_not_advance_step_count() -> None:
    env = MockEnvironment()
    env.reset(_make_subtask())
    env.observe()
    env.observe()
    assert env.step_count == 0


def test_observe_returns_valid_observation() -> None:
    env = MockEnvironment(state_dim=4)
    env.reset(_make_subtask())
    obs = env.observe()
    assert obs.state.shape == (4,)
    assert np.all(np.isfinite(obs.state))
