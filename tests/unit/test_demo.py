"""Unit tests for the M8 portfolio demo functions.

run_replay_demo() — no [agent] extra needed; always runs.
run_mock_agent_demo() — requires [agent]; skipped gracefully when absent.
run_simulation_demo() — requires [agent]; skipped gracefully when absent.
"""

import pytest

from langgraph_vla_agent.demo import run_replay_demo

# ---------------------------------------------------------------------------
# Offline/replay demo (no extras needed)
# ---------------------------------------------------------------------------


def test_replay_demo_returns_dict() -> None:
    result = run_replay_demo()
    assert isinstance(result, dict)


def test_replay_demo_mode_key() -> None:
    result = run_replay_demo()
    assert result["mode"] == "replay"


def test_replay_demo_episode_count() -> None:
    result = run_replay_demo()
    assert result["n_episodes"] == 3


def test_replay_demo_step_count_positive() -> None:
    result = run_replay_demo()
    assert result["n_steps"] > 0


def test_replay_demo_l1_mean_nonnegative() -> None:
    result = run_replay_demo()
    assert result["l1_mean"] >= 0.0


def test_replay_demo_l1_std_nonnegative() -> None:
    result = run_replay_demo()
    assert result["l1_std"] >= 0.0


def test_replay_demo_l2_mean_nonnegative() -> None:
    result = run_replay_demo()
    assert result["l2_mean"] >= 0.0


def test_replay_demo_l1_nonzero_for_mock_policy() -> None:
    """MockRobotPolicy returns zeros; ground-truth actions are nonzero → L1 > 0."""
    result = run_replay_demo()
    assert result["l1_mean"] > 0.0


def test_replay_demo_reproducible() -> None:
    """Two calls return identical metrics (deterministic policy + fixtures)."""
    r1 = run_replay_demo()
    r2 = run_replay_demo()
    assert r1["l1_mean"] == r2["l1_mean"]
    assert r1["n_steps"] == r2["n_steps"]


# ---------------------------------------------------------------------------
# Mock agent demo (requires [agent] extra)
# ---------------------------------------------------------------------------

pytest.importorskip("langgraph", reason="[agent] extra not installed")

from langgraph_vla_agent.demo import run_mock_agent_demo  # noqa: E402


def test_mock_agent_demo_returns_dict() -> None:
    result = run_mock_agent_demo()
    assert isinstance(result, dict)


def test_mock_agent_demo_mode_key() -> None:
    result = run_mock_agent_demo()
    assert result["mode"] == "mock"


def test_mock_agent_demo_completes() -> None:
    result = run_mock_agent_demo()
    assert result["final_status"] == "completed"


def test_mock_agent_demo_subtask_count() -> None:
    """Coarse granularity → 2 subtasks planned and completed."""
    result = run_mock_agent_demo()
    assert result["n_subtasks_planned"] == 2
    assert result["n_subtasks_completed"] == 2


def test_mock_agent_demo_policy_calls_match_subtasks() -> None:
    """On a clean success (no retries), policy calls = subtasks planned."""
    result = run_mock_agent_demo()
    assert result["n_policy_calls"] == result["n_subtasks_planned"]


def test_mock_agent_demo_no_retries_on_clean_success() -> None:
    result = run_mock_agent_demo()
    assert result["retry_count"] == 0
    assert result["replan_count"] == 0


# ---------------------------------------------------------------------------
# Simulation demo (requires [agent] extra)
# ---------------------------------------------------------------------------

from langgraph_vla_agent.demo import run_simulation_demo  # noqa: E402


def test_simulation_demo_returns_dict() -> None:
    result = run_simulation_demo()
    assert isinstance(result, dict)


def test_simulation_demo_mode_key() -> None:
    result = run_simulation_demo()
    assert result["mode"] == "simulation"


def test_simulation_demo_vla_only_fails_on_hard_scenario() -> None:
    result = run_simulation_demo()
    assert result["vla_rate"] == pytest.approx(0.0)


def test_simulation_demo_coarse_succeeds_on_hard_scenario() -> None:
    result = run_simulation_demo()
    assert result["coarse_rate"] == pytest.approx(1.0)


def test_simulation_demo_fine_succeeds_on_hard_scenario() -> None:
    result = run_simulation_demo()
    assert result["fine_rate"] == pytest.approx(1.0)


def test_simulation_demo_vla_rate_strictly_lower() -> None:
    result = run_simulation_demo()
    assert result["vla_rate"] < result["coarse_rate"]
    assert result["vla_rate"] < result["fine_rate"]


def test_simulation_demo_scenario_keys() -> None:
    result = run_simulation_demo()
    sc = result["scenario"]
    assert "total_progress" in sc
    assert "progress_per_step" in sc
    assert "max_steps_per_subtask" in sc
