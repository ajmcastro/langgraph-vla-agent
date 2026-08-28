"""Unit tests for SimulationEnvironment and SimulationScenario.

All tests are closed-loop: actions affect world state and the success predicate.
No LangGraph or [agent] extra is required.
"""

import numpy as np
import pytest

from langgraph_vla_agent.domain.actions import RobotAction
from langgraph_vla_agent.domain.tasks import SubTask
from langgraph_vla_agent.environments.base import RobotEnvironment
from langgraph_vla_agent.environments.simulation import SimulationEnvironment, SimulationScenario

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _subtask(instruction: str = "pick up the cube") -> SubTask:
    return SubTask(instruction=instruction)


def _zero_action(dim: int = 6) -> RobotAction:
    return RobotAction(values=np.zeros(dim, dtype=np.float32))


def _positive_action(dim: int = 6, value: float = 1.0) -> RobotAction:
    return RobotAction(values=np.full(dim, value, dtype=np.float32))


def _negative_action(dim: int = 6, value: float = -1.0) -> RobotAction:
    return RobotAction(values=np.full(dim, value, dtype=np.float32))


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


def test_simulation_environment_satisfies_protocol() -> None:
    env = SimulationEnvironment()
    assert isinstance(env, RobotEnvironment)


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


def test_progress_starts_at_zero() -> None:
    env = SimulationEnvironment()
    assert env.progress == 0.0


def test_step_count_starts_at_zero() -> None:
    env = SimulationEnvironment()
    assert env.step_count == 0


def test_reset_returns_observation_with_zero_progress() -> None:
    env = SimulationEnvironment()
    obs = env.reset(_subtask())
    assert obs.state[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Progress mechanics
# ---------------------------------------------------------------------------


def test_zero_action_makes_half_speed_progress() -> None:
    """Zero actions → action_contribution=0.5 → delta=0.5*progress_per_step."""
    env = SimulationEnvironment(success_threshold=1.0, progress_per_step=0.2, noise_scale=0.0)
    env.reset(_subtask())
    env.step(_zero_action())
    # contribution = (0 + 1)/2 = 0.5, delta = 0.5 * 0.2 = 0.1
    assert env.progress == pytest.approx(0.1, abs=1e-6)


def test_positive_action_makes_full_speed_progress() -> None:
    """All-ones actions → contribution=1.0 → delta=1.0*progress_per_step."""
    env = SimulationEnvironment(success_threshold=1.0, progress_per_step=0.2, noise_scale=0.0)
    env.reset(_subtask())
    env.step(_positive_action())
    assert env.progress == pytest.approx(0.2, abs=1e-6)


def test_negative_action_makes_zero_speed_progress() -> None:
    """All-minus-one actions → contribution=0.0 → delta=0."""
    env = SimulationEnvironment(success_threshold=1.0, progress_per_step=0.2, noise_scale=0.0)
    env.reset(_subtask())
    env.step(_negative_action())
    assert env.progress == pytest.approx(0.0, abs=1e-6)


def test_progress_accumulates_across_steps() -> None:
    env = SimulationEnvironment(success_threshold=1.0, progress_per_step=0.2, noise_scale=0.0)
    env.reset(_subtask())
    for _ in range(3):
        env.step(_zero_action())
    # 3 steps x 0.1 per step = 0.3
    assert env.progress == pytest.approx(0.3, abs=1e-6)


def test_progress_clipped_at_one() -> None:
    env = SimulationEnvironment(success_threshold=2.0, progress_per_step=0.6, noise_scale=0.0)
    env.reset(_subtask())
    env.step(_positive_action())  # delta = 0.6, progress = 0.6
    env.step(_positive_action())  # delta = 0.6, but clipped to 1.0
    assert env.progress == pytest.approx(1.0, abs=1e-6)


def test_progress_clipped_at_zero_with_noise() -> None:
    """Even with negative noise, progress never goes below 0."""
    env = SimulationEnvironment(
        success_threshold=1.0, progress_per_step=0.01, noise_scale=1.0, seed=0
    )
    env.reset(_subtask())
    for _ in range(20):
        env.step(_negative_action())
    assert env.progress >= 0.0


# ---------------------------------------------------------------------------
# Success predicate
# ---------------------------------------------------------------------------


def test_step_not_terminated_before_threshold() -> None:
    env = SimulationEnvironment(success_threshold=0.5, progress_per_step=0.1, noise_scale=0.0)
    env.reset(_subtask())
    _, result = env.step(_zero_action())  # progress = 0.05
    assert not result.terminated
    assert not result.success


def test_step_terminated_at_threshold() -> None:
    """Episode terminates with success=True when progress >= threshold."""
    env = SimulationEnvironment(success_threshold=0.1, progress_per_step=0.2, noise_scale=0.0)
    env.reset(_subtask())
    _, result = env.step(_zero_action())  # progress = 0.1, threshold = 0.1
    assert result.terminated
    assert result.success


def test_success_requires_reaching_threshold() -> None:
    env = SimulationEnvironment(success_threshold=0.5, progress_per_step=0.15, noise_scale=0.0)
    env.reset(_subtask())
    results = [env.step(_zero_action())[1] for _ in range(10)]
    # With zero actions, delta = 0.075/step; threshold=0.5 → needs 7 steps
    assert not results[5].success  # step 6: 0.45 < 0.5
    assert results[6].success  # step 7: 0.525 >= 0.5


# ---------------------------------------------------------------------------
# Step count
# ---------------------------------------------------------------------------


def test_step_count_increments() -> None:
    env = SimulationEnvironment()
    env.reset(_subtask())
    for i in range(1, 4):
        env.step(_zero_action())
        assert env.step_count == i


def test_reset_clears_step_count() -> None:
    env = SimulationEnvironment()
    env.reset(_subtask())
    env.step(_zero_action())
    env.step(_zero_action())
    env.reset(_subtask())
    assert env.step_count == 0


# ---------------------------------------------------------------------------
# Observation encoding
# ---------------------------------------------------------------------------


def test_observation_state_encodes_progress() -> None:
    env = SimulationEnvironment(success_threshold=1.0, progress_per_step=0.2, noise_scale=0.0)
    env.reset(_subtask())
    env.step(_zero_action())  # progress = 0.1
    obs = env.observe()
    assert obs.state[0] == pytest.approx(env.progress, abs=1e-6)


def test_observe_does_not_advance_step_count() -> None:
    env = SimulationEnvironment()
    env.reset(_subtask())
    env.observe()
    assert env.step_count == 0


def test_step_result_info_contains_progress() -> None:
    env = SimulationEnvironment(success_threshold=1.0, progress_per_step=0.2, noise_scale=0.0)
    env.reset(_subtask())
    _, result = env.step(_zero_action())
    assert "progress" in result.info
    assert result.info["progress"] >= 0.0


# ---------------------------------------------------------------------------
# Reset reproduciblity
# ---------------------------------------------------------------------------


def test_reset_rewinds_progress() -> None:
    env = SimulationEnvironment(success_threshold=1.0, progress_per_step=0.2, noise_scale=0.0)
    env.reset(_subtask())
    env.step(_zero_action())
    env.step(_zero_action())
    env.reset(_subtask())
    assert env.progress == pytest.approx(0.0)


def test_two_resets_give_identical_first_step() -> None:
    env = SimulationEnvironment(success_threshold=1.0, progress_per_step=0.2, noise_scale=0.0)
    env.reset(_subtask())
    env.step(_zero_action())
    p1 = env.progress

    env.reset(_subtask())
    env.step(_zero_action())
    p2 = env.progress

    assert p1 == pytest.approx(p2)


# ---------------------------------------------------------------------------
# SimulationScenario validation
# ---------------------------------------------------------------------------


def test_simulation_scenario_defaults_are_valid() -> None:
    s = SimulationScenario()
    assert 0 < s.total_progress <= 1.0
    assert s.progress_per_step > 0
    assert s.max_steps_per_subtask > 0
    assert s.noise_scale >= 0.0


def test_simulation_scenario_rejects_zero_total_progress() -> None:
    with pytest.raises(ValueError):
        SimulationScenario(total_progress=0.0)


def test_simulation_scenario_rejects_negative_progress_per_step() -> None:
    with pytest.raises(ValueError):
        SimulationScenario(progress_per_step=-0.1)
