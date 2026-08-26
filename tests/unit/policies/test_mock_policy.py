"""Unit tests for MockRobotPolicy."""

import numpy as np
import pytest

from langgraph_vla_agent.domain import EvaluationMode, PolicyContext, RobotObservation
from langgraph_vla_agent.policies import MockPolicyBehavior, MockRobotPolicy, RobotPolicy


def _make_context(seed: int | None = None) -> PolicyContext:
    return PolicyContext(
        run_id="run-test",
        episode_id="ep-test",
        evaluation_mode=EvaluationMode.MOCK,
        seed=seed,
    )


def _make_obs(state_dim: int = 6) -> RobotObservation:
    return RobotObservation(state=np.zeros(state_dim, dtype=np.float32))


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_mock_policy_satisfies_protocol() -> None:
    policy = MockRobotPolicy()
    assert isinstance(policy, RobotPolicy)


# ---------------------------------------------------------------------------
# ALWAYS_VALID behavior
# ---------------------------------------------------------------------------


def test_always_valid_returns_finite_zeros() -> None:
    policy = MockRobotPolicy(behavior=MockPolicyBehavior.ALWAYS_VALID, action_dim=6)
    policy.reset(_make_context())
    action = policy.act(_make_obs(), "grasp the cube")
    assert action.values.shape == (6,)
    assert np.all(np.isfinite(action.values))
    assert np.all(action.values == 0.0)


def test_always_valid_never_returns_nan() -> None:
    policy = MockRobotPolicy(behavior=MockPolicyBehavior.ALWAYS_VALID, action_dim=6, n=1)
    policy.reset(_make_context())
    for _ in range(10):
        action = policy.act(_make_obs(), "test")
        assert np.all(np.isfinite(action.values))


# ---------------------------------------------------------------------------
# INVALID_AFTER_N behavior
# ---------------------------------------------------------------------------


def test_invalid_after_n_returns_nan_at_threshold() -> None:
    policy = MockRobotPolicy(behavior=MockPolicyBehavior.INVALID_AFTER_N, action_dim=6, n=3)
    policy.reset(_make_context())
    obs = _make_obs()

    for _ in range(3):
        action = policy.act(obs, "test")
        assert np.all(np.isfinite(action.values)), "Steps 1-3 should be valid"

    action = policy.act(obs, "test")  # step 4
    assert not np.all(np.isfinite(action.values)), "Step 4 should be NaN (invalid)"


# ---------------------------------------------------------------------------
# RAISE_AFTER_N behavior
# ---------------------------------------------------------------------------


def test_raise_after_n_raises_at_threshold() -> None:
    policy = MockRobotPolicy(behavior=MockPolicyBehavior.RAISE_AFTER_N, action_dim=6, n=2)
    policy.reset(_make_context())
    obs = _make_obs()

    policy.act(obs, "test")  # step 1 — ok
    policy.act(obs, "test")  # step 2 — ok

    with pytest.raises(RuntimeError, match="simulated policy crash"):
        policy.act(obs, "test")  # step 3 — raises


# ---------------------------------------------------------------------------
# reset() semantics
# ---------------------------------------------------------------------------


def test_reset_clears_step_count() -> None:
    policy = MockRobotPolicy(behavior=MockPolicyBehavior.INVALID_AFTER_N, n=1)
    policy.reset(_make_context())
    policy.act(_make_obs(), "test")
    policy.act(_make_obs(), "test")  # step 2 → NaN

    policy.reset(_make_context())  # should start fresh
    assert policy.step_count == 0
    action = policy.act(_make_obs(), "test")  # step 1 again → valid
    assert np.all(np.isfinite(action.values))


def test_reset_stores_context() -> None:
    policy = MockRobotPolicy()
    ctx = _make_context(seed=99)
    policy.reset(ctx)
    assert policy.last_context is ctx


def test_step_count_increments() -> None:
    policy = MockRobotPolicy()
    policy.reset(_make_context())
    assert policy.step_count == 0
    policy.act(_make_obs(), "a")
    policy.act(_make_obs(), "b")
    assert policy.step_count == 2
