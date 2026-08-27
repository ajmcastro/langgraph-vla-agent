"""Unit tests for ReplayEnvironment — including full Executor integration."""

import pathlib

import numpy as np
import pytest

from langgraph_vla_agent.datasets.episode import ReplayEpisode, ReplayStep
from langgraph_vla_agent.datasets.store import FixtureEpisodeStore
from langgraph_vla_agent.domain.actions import RobotAction
from langgraph_vla_agent.domain.context import EvaluationMode, PolicyContext
from langgraph_vla_agent.domain.results import ExecutionStatus
from langgraph_vla_agent.domain.tasks import SubTask
from langgraph_vla_agent.environments import ReplayEnvironment, RobotEnvironment
from langgraph_vla_agent.execution import Executor, ExecutorConfig
from langgraph_vla_agent.policies import ReplayRobotPolicy

_FIXTURES_DIR = pathlib.Path(__file__).parents[3] / "data" / "fixtures" / "episodes"


def _make_episode(n_steps: int = 3, success: bool = True) -> ReplayEpisode:
    steps = [
        ReplayStep(
            timestep=i,
            observation={"state": [float(i)] + [0.0] * 5},
            action=[0.1] * 6,
            terminated=(i == n_steps - 1),
            success=(success and i == n_steps - 1),
        )
        for i in range(n_steps)
    ]
    return ReplayEpisode(
        episode_id="ep-env-test",
        instruction="test",
        dataset_id="fixture",
        action_dim=6,
        state_dim=6,
        steps=steps,
    )


def _action() -> RobotAction:
    return RobotAction(values=np.zeros(6, dtype=np.float32))


def _subtask(ep: ReplayEpisode) -> SubTask:
    return SubTask(id="st-1", instruction=ep.instruction)


def _context(mode: EvaluationMode = EvaluationMode.REPLAY) -> PolicyContext:
    return PolicyContext(run_id="r", episode_id="e", evaluation_mode=mode)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_replay_environment_satisfies_protocol() -> None:
    env = ReplayEnvironment(_make_episode())
    assert isinstance(env, RobotEnvironment)


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


def test_replay_environment_reset_returns_first_obs() -> None:
    ep = _make_episode(n_steps=3)
    env = ReplayEnvironment(ep)
    obs = env.reset(_subtask(ep))
    # First step has state=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert obs.state[0] == pytest.approx(0.0)
    assert obs.timestamp == pytest.approx(0.0)


def test_replay_environment_reset_rewinds_ptr() -> None:
    ep = _make_episode(n_steps=3)
    env = ReplayEnvironment(ep)
    env.reset(_subtask(ep))
    env.step(_action())
    assert env.ptr == 1

    env.reset(_subtask(ep))
    assert env.ptr == 0


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------


def test_replay_environment_step_advances_ptr() -> None:
    ep = _make_episode(n_steps=3)
    env = ReplayEnvironment(ep)
    env.reset(_subtask(ep))
    env.step(_action())
    assert env.ptr == 1


def test_replay_environment_step_returns_next_obs_state() -> None:
    ep = _make_episode(n_steps=3)
    env = ReplayEnvironment(ep)
    env.reset(_subtask(ep))
    # step() from ptr=0 → returns obs of step[1] (state[0]=1.0) and result of step[0]
    next_obs, _ = env.step(_action())
    assert next_obs.state[0] == pytest.approx(1.0)


def test_replay_environment_step_returns_success_on_terminal() -> None:
    ep = _make_episode(n_steps=3, success=True)
    env = ReplayEnvironment(ep)
    env.reset(_subtask(ep))
    env.step(_action())  # step 0
    env.step(_action())  # step 1
    _, result = env.step(_action())  # step 2 — terminal
    assert result.terminated is True
    assert result.success is True


def test_replay_environment_step_returns_failure_on_failed_episode() -> None:
    ep = _make_episode(n_steps=3, success=False)
    env = ReplayEnvironment(ep)
    env.reset(_subtask(ep))
    env.step(_action())
    env.step(_action())
    _, result = env.step(_action())
    assert result.terminated is True
    assert result.success is False


def test_replay_environment_overrun_truncates() -> None:
    ep = _make_episode(n_steps=2)
    env = ReplayEnvironment(ep)
    env.reset(_subtask(ep))
    env.step(_action())  # step 0
    env.step(_action())  # step 1 — terminal
    _, result = env.step(_action())  # over-run
    assert result.truncated is True


def test_replay_environment_action_is_ignored() -> None:
    """The environment serves recorded observations regardless of the action."""
    ep = _make_episode(n_steps=3)
    env1 = ReplayEnvironment(ep)
    env2 = ReplayEnvironment(ep)
    env1.reset(_subtask(ep))
    env2.reset(_subtask(ep))

    zero_action = RobotAction(values=np.zeros(6, dtype=np.float32))
    ones_action = RobotAction(values=np.ones(6, dtype=np.float32))

    obs1, _ = env1.step(zero_action)
    obs2, _ = env2.step(ones_action)

    np.testing.assert_array_equal(obs1.state, obs2.state)


# ---------------------------------------------------------------------------
# observe()
# ---------------------------------------------------------------------------


def test_replay_environment_observe_does_not_advance_ptr() -> None:
    ep = _make_episode(n_steps=3)
    env = ReplayEnvironment(ep)
    env.reset(_subtask(ep))
    env.observe()
    env.observe()
    assert env.ptr == 0


# ---------------------------------------------------------------------------
# Executor integration — full replay loop end-to-end
# ---------------------------------------------------------------------------


def test_executor_replay_success_episode() -> None:
    store = FixtureEpisodeStore(_FIXTURES_DIR)
    ep = store.load_episode("fixture_episode_001")
    policy = ReplayRobotPolicy(ep)
    env = ReplayEnvironment(ep)
    config = ExecutorConfig(evaluation_mode=EvaluationMode.REPLAY, max_steps=20)
    ctx = PolicyContext(run_id="r", episode_id="e", evaluation_mode=EvaluationMode.REPLAY)
    subtask = SubTask(id="st-1", instruction=ep.instruction)

    result = Executor(policy, env, config).run(subtask, ctx)

    assert result.status == ExecutionStatus.SUCCESS
    assert result.steps_taken == ep.length
    assert result.evaluation_mode == EvaluationMode.REPLAY
    assert result.succeeded


def test_executor_replay_failure_episode() -> None:
    store = FixtureEpisodeStore(_FIXTURES_DIR)
    ep = store.load_episode("fixture_episode_002")
    policy = ReplayRobotPolicy(ep)
    env = ReplayEnvironment(ep)
    config = ExecutorConfig(evaluation_mode=EvaluationMode.REPLAY, max_steps=20)
    ctx = PolicyContext(run_id="r", episode_id="e", evaluation_mode=EvaluationMode.REPLAY)
    subtask = SubTask(id="st-2", instruction=ep.instruction)

    result = Executor(policy, env, config).run(subtask, ctx)

    assert result.status == ExecutionStatus.FAILURE
    assert result.steps_taken == ep.length
    assert not result.succeeded


def test_executor_replay_result_contains_elapsed_metric() -> None:
    store = FixtureEpisodeStore(_FIXTURES_DIR)
    ep = store.load_episode("fixture_episode_003")
    result = Executor(
        ReplayRobotPolicy(ep),
        ReplayEnvironment(ep),
        ExecutorConfig(evaluation_mode=EvaluationMode.REPLAY, max_steps=20),
    ).run(
        SubTask(id="st-3", instruction=ep.instruction),
        PolicyContext(run_id="r", episode_id="e", evaluation_mode=EvaluationMode.REPLAY),
    )
    assert "elapsed_s" in result.metrics
    assert result.metrics["elapsed_s"] >= 0.0
