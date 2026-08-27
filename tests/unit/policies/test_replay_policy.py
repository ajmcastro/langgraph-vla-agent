"""Unit tests for ReplayRobotPolicy."""

import numpy as np
import pytest

from langgraph_vla_agent.datasets.episode import ReplayEpisode, ReplayStep
from langgraph_vla_agent.domain.context import EvaluationMode, PolicyContext
from langgraph_vla_agent.domain.observations import RobotObservation
from langgraph_vla_agent.policies import ReplayRobotPolicy, RobotPolicy


def _make_episode(n_steps: int = 4) -> ReplayEpisode:
    """Synthetic episode with distinct action values per step."""
    steps = [
        ReplayStep(
            timestep=i,
            observation={"state": [0.0] * 6},
            action=[float(i)] + [0.0] * 5,
            terminated=(i == n_steps - 1),
            success=(i == n_steps - 1),
        )
        for i in range(n_steps)
    ]
    return ReplayEpisode(
        episode_id="ep-replay-test",
        instruction="move the block",
        dataset_id="fixture",
        action_dim=6,
        state_dim=6,
        steps=steps,
    )


def _obs() -> RobotObservation:
    return RobotObservation(state=np.zeros(6, dtype=np.float32))


def _context() -> PolicyContext:
    return PolicyContext(run_id="r", episode_id="e", evaluation_mode=EvaluationMode.REPLAY)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_replay_policy_satisfies_protocol() -> None:
    policy = ReplayRobotPolicy(_make_episode())
    assert isinstance(policy, RobotPolicy)


# ---------------------------------------------------------------------------
# Action replay order
# ---------------------------------------------------------------------------


def test_replay_policy_returns_actions_in_order() -> None:
    ep = _make_episode(n_steps=4)
    policy = ReplayRobotPolicy(ep)
    policy.reset(_context())

    for i in range(4):
        action = policy.act(_obs(), "irrelevant")
        assert action.values[0] == pytest.approx(float(i)), f"step {i}"


def test_replay_policy_action_has_correct_dim() -> None:
    policy = ReplayRobotPolicy(_make_episode())
    policy.reset(_context())
    action = policy.act(_obs(), "irrelevant")
    assert action.values.shape == (6,)


def test_replay_policy_ptr_advances_after_act() -> None:
    policy = ReplayRobotPolicy(_make_episode(n_steps=3))
    policy.reset(_context())
    assert policy.ptr == 0
    policy.act(_obs(), "irrelevant")
    assert policy.ptr == 1
    policy.act(_obs(), "irrelevant")
    assert policy.ptr == 2


# ---------------------------------------------------------------------------
# Reset semantics
# ---------------------------------------------------------------------------


def test_replay_policy_resets_ptr_to_zero() -> None:
    ep = _make_episode(n_steps=3)
    policy = ReplayRobotPolicy(ep)
    policy.reset(_context())
    policy.act(_obs(), "irrelevant")
    policy.act(_obs(), "irrelevant")
    assert policy.ptr == 2

    policy.reset(_context())
    assert policy.ptr == 0


def test_replay_policy_second_run_starts_from_beginning() -> None:
    ep = _make_episode(n_steps=2)
    policy = ReplayRobotPolicy(ep)

    policy.reset(_context())
    a1_run1 = policy.act(_obs(), "irrelevant")

    policy.reset(_context())
    a1_run2 = policy.act(_obs(), "irrelevant")

    assert a1_run1.values[0] == pytest.approx(a1_run2.values[0])


# ---------------------------------------------------------------------------
# Exhaustion guard
# ---------------------------------------------------------------------------


def test_replay_policy_raises_when_exhausted() -> None:
    ep = _make_episode(n_steps=2)
    policy = ReplayRobotPolicy(ep)
    policy.reset(_context())
    policy.act(_obs(), "irrelevant")
    policy.act(_obs(), "irrelevant")

    with pytest.raises(IndexError, match="exhausted"):
        policy.act(_obs(), "irrelevant")
