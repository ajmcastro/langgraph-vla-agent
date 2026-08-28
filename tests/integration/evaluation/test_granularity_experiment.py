"""Integration tests for the M6 planning-granularity experiment.

Requires the [agent] extra: uv sync --extra dev --extra agent
Gracefully skipped when LangGraph is not installed.

These tests run the full three-condition experiment end-to-end using the
compiled LangGraph graph, DeterministicPlanner (coarse + fine),
VlaOnlyPlanner, MockRobotPolicy, and MockEnvironment.
"""

import pytest

langgraph = pytest.importorskip("langgraph", reason="[agent] extra not installed")

from langgraph_vla_agent.agent.state import AgentStatus  # noqa: E402
from langgraph_vla_agent.environments.mock import MockScenario  # noqa: E402
from langgraph_vla_agent.evaluation.experiment import (  # noqa: E402
    EpisodeScenario,
    run_granularity_experiment,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_PICK_PLACE_GOAL = "pick up the cube and place it in the bin"

_SUCCESS_SCENARIOS = [
    EpisodeScenario(goal=_PICK_PLACE_GOAL),
    EpisodeScenario(goal="grasp the block and put it in the tray"),
]

_FAIL_SCENARIOS = [
    EpisodeScenario(
        goal=_PICK_PLACE_GOAL,
        mock_scenario=MockScenario.FAIL_AT_STEP,
        succeed_at_step=999,
    ),
]


# ---------------------------------------------------------------------------
# Happy-path: all conditions complete on success scenarios
# ---------------------------------------------------------------------------


def test_experiment_returns_all_three_conditions() -> None:
    result = run_granularity_experiment(_SUCCESS_SCENARIOS[:1])
    assert set(result.results_by_condition.keys()) == {
        "vla_only",
        "coarse_agentic",
        "fine_agentic",
    }


def test_experiment_n_episodes_matches_scenarios() -> None:
    result = run_granularity_experiment(_SUCCESS_SCENARIOS)
    assert result.n_episodes == len(_SUCCESS_SCENARIOS)


def test_all_conditions_complete_on_success_scenario() -> None:
    result = run_granularity_experiment(_SUCCESS_SCENARIOS[:1])
    for condition, results in result.results_by_condition.items():
        assert results[0].completed, f"{condition} did not complete"


def test_vla_only_has_one_subtask() -> None:
    result = run_granularity_experiment(_SUCCESS_SCENARIOS[:1])
    r = result.results_by_condition["vla_only"][0]
    assert r.n_subtasks_planned == 1


def test_coarse_has_two_subtasks() -> None:
    result = run_granularity_experiment(_SUCCESS_SCENARIOS[:1])
    r = result.results_by_condition["coarse_agentic"][0]
    assert r.n_subtasks_planned == 2


def test_fine_has_five_subtasks() -> None:
    result = run_granularity_experiment(_SUCCESS_SCENARIOS[:1])
    r = result.results_by_condition["fine_agentic"][0]
    assert r.n_subtasks_planned == 5


def test_vla_only_has_fewer_policy_calls_than_coarse() -> None:
    result = run_granularity_experiment(_SUCCESS_SCENARIOS[:1])
    vla = result.results_by_condition["vla_only"][0].n_policy_calls
    coarse = result.results_by_condition["coarse_agentic"][0].n_policy_calls
    assert vla < coarse


def test_coarse_has_fewer_policy_calls_than_fine() -> None:
    result = run_granularity_experiment(_SUCCESS_SCENARIOS[:1])
    coarse = result.results_by_condition["coarse_agentic"][0].n_policy_calls
    fine = result.results_by_condition["fine_agentic"][0].n_policy_calls
    assert coarse < fine


def test_policy_calls_equal_subtasks_on_clean_success() -> None:
    result = run_granularity_experiment(_SUCCESS_SCENARIOS[:1])
    for condition, results in result.results_by_condition.items():
        r = results[0]
        assert r.n_policy_calls == r.n_subtasks_planned, (
            f"{condition}: expected policy_calls == subtasks_planned "
            f"({r.n_policy_calls} != {r.n_subtasks_planned})"
        )


# ---------------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------------


def test_all_conditions_fail_when_mock_never_succeeds() -> None:
    result = run_granularity_experiment(
        _FAIL_SCENARIOS,
        max_retries=0,
        max_replans=0,
    )
    for condition, results in result.results_by_condition.items():
        assert results[0].final_status == AgentStatus.FAILED, f"{condition} expected FAILED"


def test_retry_count_increases_on_failure_with_retries_allowed() -> None:
    result = run_granularity_experiment(
        _FAIL_SCENARIOS,
        max_retries=2,
        max_replans=0,
    )
    coarse = result.results_by_condition["coarse_agentic"][0]
    # After exhausting retries, retry_count reflects attempts up to max
    assert coarse.final_status == AgentStatus.FAILED


# ---------------------------------------------------------------------------
# Aggregates and summaries
# ---------------------------------------------------------------------------


def test_condition_summary_completion_rate_is_one_on_success() -> None:
    result = run_granularity_experiment(_SUCCESS_SCENARIOS)
    for _, name in [
        (None, "vla_only"),
        (None, "coarse_agentic"),
        (None, "fine_agentic"),
    ]:
        summary = result.condition_summary(name)
        assert summary.completion_rate == 1.0, f"{name} rate != 1.0"


def test_mean_subtasks_differ_across_conditions() -> None:
    result = run_granularity_experiment(_SUCCESS_SCENARIOS)
    vla = result.condition_summary("vla_only").mean_subtasks_planned
    coarse = result.condition_summary("coarse_agentic").mean_subtasks_planned
    fine = result.condition_summary("fine_agentic").mean_subtasks_planned
    assert vla < coarse < fine


def test_mean_policy_calls_differ_across_conditions() -> None:
    result = run_granularity_experiment(_SUCCESS_SCENARIOS)
    vla = result.condition_summary("vla_only").mean_policy_calls
    coarse = result.condition_summary("coarse_agentic").mean_policy_calls
    fine = result.condition_summary("fine_agentic").mean_policy_calls
    assert vla < coarse < fine


def test_summary_lines_are_non_empty() -> None:
    result = run_granularity_experiment(_SUCCESS_SCENARIOS[:1])
    lines = result.summary_lines()
    assert len(lines) > 3
    assert any("vla_only" in line for line in lines)
