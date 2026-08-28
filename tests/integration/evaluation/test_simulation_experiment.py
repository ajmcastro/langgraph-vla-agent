"""Integration tests for the M7 simulation-mode planning-granularity experiment.

Requires the [agent] extra: uv sync --extra dev --extra agent
Gracefully skipped when LangGraph is not installed.

These tests run the full three-condition experiment end-to-end using the
compiled LangGraph graph, DeterministicPlanner (coarse + fine),
VlaOnlyPlanner, MockRobotPolicy, and SimulationEnvironment.

Key assertions:
- Easy scenario: all three conditions complete (same as mock mode).
- Hard scenario: vla_only FAILS, agentic conditions SUCCEED.
  This is the result mock mode could not produce.
- completion_rate(vla_only) < completion_rate(coarse) on hard scenarios.
"""

import pytest

langgraph = pytest.importorskip("langgraph", reason="[agent] extra not installed")

from langgraph_vla_agent.agent.state import AgentStatus  # noqa: E402
from langgraph_vla_agent.environments.simulation import SimulationScenario  # noqa: E402
from langgraph_vla_agent.evaluation.simulation import (  # noqa: E402
    SimulationEpisodeScenario,
    run_simulation_experiment,
)

# ---------------------------------------------------------------------------
# Scenario fixtures
# ---------------------------------------------------------------------------

_GOAL = "pick up the cube and place it in the bin"

# Easy: zero-action baseline reaches threshold in < max_steps for all conditions.
# threshold for vla_only = 0.3 / 1 = 0.3; steps needed ≈ ceil(0.3/0.075) = 4
_EASY_SCENARIO = SimulationScenario(
    total_progress=0.3,
    progress_per_step=0.15,
    max_steps_per_subtask=10,
    noise_scale=0.0,
    seed=42,
)

# Hard: vla_only threshold = 0.5 / 1 = 0.5; steps needed ≈ ceil(0.5/0.075) = 7 > 5
#       coarse threshold = 0.5 / 2 = 0.25; steps needed ≈ 4 <= 5 → succeeds
#       fine threshold   = 0.5 / 5 = 0.1; steps needed ≈ 2 <= 5 → succeeds
_HARD_SCENARIO = SimulationScenario(
    total_progress=0.5,
    progress_per_step=0.15,
    max_steps_per_subtask=5,
    noise_scale=0.0,
    seed=42,
)

_EASY_EPISODES = [
    SimulationEpisodeScenario(goal=_GOAL, scenario=_EASY_SCENARIO),
    SimulationEpisodeScenario(
        goal="grasp the block and put it in the tray", scenario=_EASY_SCENARIO
    ),
]

_HARD_EPISODES = [
    SimulationEpisodeScenario(goal=_GOAL, scenario=_HARD_SCENARIO),
    SimulationEpisodeScenario(
        goal="pick up the red block and put it in the tray", scenario=_HARD_SCENARIO
    ),
]


# ---------------------------------------------------------------------------
# Easy scenario — all conditions succeed (mirrors mock mode)
# ---------------------------------------------------------------------------


def test_experiment_returns_all_three_conditions() -> None:
    result = run_simulation_experiment(_EASY_EPISODES[:1])
    assert set(result.results_by_condition.keys()) == {
        "vla_only",
        "coarse_agentic",
        "fine_agentic",
    }


def test_experiment_n_episodes_matches_scenarios() -> None:
    result = run_simulation_experiment(_EASY_EPISODES)
    assert result.n_episodes == len(_EASY_EPISODES)


def test_all_conditions_complete_on_easy_scenario() -> None:
    result = run_simulation_experiment(_EASY_EPISODES[:1])
    for condition, results in result.results_by_condition.items():
        assert results[0].completed, f"{condition} did not complete on easy scenario"


def test_easy_completion_rates_are_one() -> None:
    result = run_simulation_experiment(_EASY_EPISODES)
    for _, name in [
        (None, "vla_only"),
        (None, "coarse_agentic"),
        (None, "fine_agentic"),
    ]:
        summary = result.condition_summary(name)
        assert summary.completion_rate == pytest.approx(1.0), f"{name} rate != 1.0 on easy"


# ---------------------------------------------------------------------------
# Hard scenario — closed-loop differentiation (the key M7 result)
# ---------------------------------------------------------------------------


def test_vla_only_fails_on_hard_scenario() -> None:
    """VLA-only cannot reach threshold=0.5 in 5 steps → FAILED."""
    result = run_simulation_experiment(_HARD_EPISODES[:1], max_retries=0, max_replans=0)
    r = result.results_by_condition["vla_only"][0]
    assert r.final_status == AgentStatus.FAILED, (
        f"Expected vla_only to FAIL on hard scenario, got {r.final_status}"
    )


def test_coarse_succeeds_on_hard_scenario() -> None:
    """Coarse threshold=0.25 is reachable in 5 steps → COMPLETED."""
    result = run_simulation_experiment(_HARD_EPISODES[:1], max_retries=0, max_replans=0)
    r = result.results_by_condition["coarse_agentic"][0]
    assert r.completed, "Expected coarse_agentic to COMPLETE on hard scenario"


def test_fine_succeeds_on_hard_scenario() -> None:
    """Fine threshold=0.1 is reachable in 5 steps → COMPLETED."""
    result = run_simulation_experiment(_HARD_EPISODES[:1], max_retries=0, max_replans=0)
    r = result.results_by_condition["fine_agentic"][0]
    assert r.completed, "Expected fine_agentic to COMPLETE on hard scenario"


def test_hard_completion_rate_differs_across_conditions() -> None:
    """Completion rate is strictly lower for vla_only than agentic conditions."""
    result = run_simulation_experiment(_HARD_EPISODES, max_retries=0, max_replans=0)
    vla_rate = result.condition_summary("vla_only").completion_rate
    coarse_rate = result.condition_summary("coarse_agentic").completion_rate
    fine_rate = result.condition_summary("fine_agentic").completion_rate
    assert vla_rate < coarse_rate, f"Expected vla_only < coarse; got {vla_rate} vs {coarse_rate}"
    assert vla_rate < fine_rate, f"Expected vla_only < fine; got {vla_rate} vs {fine_rate}"


# ---------------------------------------------------------------------------
# Per-subtask threshold scaling
# ---------------------------------------------------------------------------


def test_per_subtask_threshold_scales_with_n_subtasks() -> None:
    result = run_simulation_experiment(_HARD_EPISODES[:1])
    vla_threshold = result.condition_summary("vla_only").per_subtask_threshold
    coarse_threshold = result.condition_summary("coarse_agentic").per_subtask_threshold
    fine_threshold = result.condition_summary("fine_agentic").per_subtask_threshold
    assert vla_threshold > coarse_threshold > fine_threshold


def test_total_progress_consistent_across_conditions() -> None:
    result = run_simulation_experiment(_HARD_EPISODES[:1])
    totals = {
        name: result.condition_summary(name).total_progress_required
        for name in ["vla_only", "coarse_agentic", "fine_agentic"]
    }
    assert len(set(totals.values())) == 1, f"Total progress differs: {totals}"


# ---------------------------------------------------------------------------
# Aggregates and summaries
# ---------------------------------------------------------------------------


def test_summary_lines_contain_all_three_conditions() -> None:
    result = run_simulation_experiment(_EASY_EPISODES[:1])
    joined = "\n".join(result.summary_lines())
    for condition in ["vla_only", "coarse_agentic", "fine_agentic"]:
        assert condition in joined


def test_summary_lines_contain_evaluation_note() -> None:
    result = run_simulation_experiment(_EASY_EPISODES[:1])
    joined = "\n".join(result.summary_lines())
    assert "simulation" in joined.lower()


def test_result_carries_scenario() -> None:
    result = run_simulation_experiment(_EASY_EPISODES[:1])
    assert result.scenario.total_progress == pytest.approx(_EASY_SCENARIO.total_progress)
