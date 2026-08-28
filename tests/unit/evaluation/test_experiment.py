"""Unit tests for GranularityExperiment models and VlaOnlyPlanner.

All tests run without LangGraph — they exercise the data models, aggregate
computations, and VlaOnlyPlanner directly without invoking the agent graph.
"""

from langgraph_vla_agent.agent.state import AgentStatus
from langgraph_vla_agent.domain.context import EvaluationMode
from langgraph_vla_agent.domain.tasks import TaskGoal
from langgraph_vla_agent.environments.mock import MockScenario
from langgraph_vla_agent.evaluation.experiment import (
    ConditionResult,
    ConditionSummary,
    EpisodeScenario,
    GranularityExperimentResult,
)
from langgraph_vla_agent.planning.vla_only import VlaOnlyPlanner

# ---------------------------------------------------------------------------
# VlaOnlyPlanner
# ---------------------------------------------------------------------------


def test_vla_only_planner_produces_single_subtask() -> None:
    goal = TaskGoal(text="pick up the cube", evaluation_mode=EvaluationMode.MOCK)
    plan = VlaOnlyPlanner().plan(goal)
    assert len(plan.subtasks) == 1


def test_vla_only_planner_uses_full_goal_as_instruction() -> None:
    goal_text = "pick up the cube and place it in the bin"
    goal = TaskGoal(text=goal_text, evaluation_mode=EvaluationMode.MOCK)
    plan = VlaOnlyPlanner().plan(goal)
    assert plan.subtasks[0].instruction == goal_text


def test_vla_only_planner_id_is_vla_only() -> None:
    goal = TaskGoal(text="pick up the cube", evaluation_mode=EvaluationMode.MOCK)
    plan = VlaOnlyPlanner().plan(goal)
    assert plan.planner_id == "vla_only"


def test_vla_only_planner_success_criteria_is_set() -> None:
    goal = TaskGoal(text="pick up the cube", evaluation_mode=EvaluationMode.MOCK)
    plan = VlaOnlyPlanner().plan(goal)
    assert plan.subtasks[0].success_criteria != ""


def test_vla_only_planner_works_with_any_goal_text() -> None:
    for goal_text in ["dance around the room", "", "pick, grasp, place"]:
        goal = TaskGoal(text=goal_text, evaluation_mode=EvaluationMode.MOCK)
        plan = VlaOnlyPlanner().plan(goal)
        assert plan.subtasks[0].instruction == goal_text


# ---------------------------------------------------------------------------
# EpisodeScenario
# ---------------------------------------------------------------------------


def test_episode_scenario_defaults() -> None:
    s = EpisodeScenario(goal="pick up the cube")
    assert s.mock_scenario == MockScenario.SUCCEED_AT_STEP
    assert s.succeed_at_step == 2


def test_episode_scenario_custom_values() -> None:
    s = EpisodeScenario(
        goal="pick up the cube",
        mock_scenario=MockScenario.FAIL_AT_STEP,
        succeed_at_step=5,
    )
    assert s.mock_scenario == MockScenario.FAIL_AT_STEP
    assert s.succeed_at_step == 5


# ---------------------------------------------------------------------------
# ConditionResult
# ---------------------------------------------------------------------------


def _make_result(
    condition: str = "coarse_agentic",
    status: str = AgentStatus.COMPLETED,
    n_planned: int = 2,
    n_completed: int = 2,
    n_policy_calls: int = 2,
) -> ConditionResult:
    return ConditionResult(
        condition=condition,
        episode_idx=0,
        goal="pick up the cube",
        final_status=status,
        n_subtasks_planned=n_planned,
        n_subtasks_completed=n_completed,
        n_subtasks_failed=0,
        n_policy_calls=n_policy_calls,
        retry_count=0,
        replan_count=0,
    )


def test_condition_result_completed_is_true_on_completed_status() -> None:
    r = _make_result(status=AgentStatus.COMPLETED)
    assert r.completed is True


def test_condition_result_completed_is_false_on_failed_status() -> None:
    r = _make_result(status=AgentStatus.FAILED)
    assert r.completed is False


def test_condition_result_completed_is_false_on_safety_stop() -> None:
    r = _make_result(status=AgentStatus.SAFETY_STOP)
    assert r.completed is False


# ---------------------------------------------------------------------------
# GranularityExperimentResult — condition_summary
# ---------------------------------------------------------------------------


def _make_experiment(
    vla: list[ConditionResult],
    coarse: list[ConditionResult],
    fine: list[ConditionResult],
) -> GranularityExperimentResult:
    return GranularityExperimentResult(
        results_by_condition={
            "vla_only": vla,
            "coarse_agentic": coarse,
            "fine_agentic": fine,
        },
        n_episodes=max(len(vla), len(coarse), len(fine)),
    )


def test_condition_summary_completion_rate_all_complete() -> None:
    results = [_make_result(status=AgentStatus.COMPLETED) for _ in range(4)]
    exp = _make_experiment(results, results, results)
    summary = exp.condition_summary("coarse_agentic")
    assert summary.completion_rate == 1.0
    assert summary.n_completed == 4


def test_condition_summary_completion_rate_none_complete() -> None:
    results = [_make_result(status=AgentStatus.FAILED) for _ in range(3)]
    exp = _make_experiment(results, results, results)
    summary = exp.condition_summary("vla_only")
    assert summary.completion_rate == 0.0
    assert summary.n_completed == 0


def test_condition_summary_mean_subtasks() -> None:
    results = [
        _make_result(n_planned=2),
        _make_result(n_planned=2),
    ]
    exp = _make_experiment(results, results, results)
    summary = exp.condition_summary("coarse_agentic")
    assert summary.mean_subtasks_planned == 2.0


def test_condition_summary_mean_policy_calls() -> None:
    results = [
        _make_result(n_policy_calls=1, condition="vla_only"),
        _make_result(n_policy_calls=1, condition="vla_only"),
    ]
    exp = _make_experiment(results, [], [])
    summary = exp.condition_summary("vla_only")
    assert summary.mean_policy_calls == 1.0


def test_condition_summary_empty_condition() -> None:
    exp = _make_experiment([], [], [])
    summary = exp.condition_summary("vla_only")
    assert summary.completion_rate == 0.0
    assert summary.n_episodes == 0


# ---------------------------------------------------------------------------
# GranularityExperimentResult — summary_lines
# ---------------------------------------------------------------------------


def test_summary_lines_has_header_and_separator() -> None:
    results = [_make_result()]
    exp = _make_experiment(results, results, results)
    lines = exp.summary_lines()
    assert len(lines) >= 5  # header, sep, 3 rows, blank, note
    assert "Condition" in lines[0]
    assert "-" * 10 in lines[1]


def test_summary_lines_has_all_three_conditions() -> None:
    results = [_make_result()]
    exp = _make_experiment(results, results, results)
    joined = "\n".join(exp.summary_lines())
    assert "vla_only" in joined
    assert "coarse_agentic" in joined
    assert "fine_agentic" in joined


def test_summary_lines_has_evaluation_note() -> None:
    results = [_make_result()]
    exp = _make_experiment(results, results, results)
    joined = "\n".join(exp.summary_lines())
    assert "mock evaluation" in joined


def test_all_summaries_returns_three_entries() -> None:
    results = [_make_result()]
    exp = _make_experiment(results, results, results)
    summaries = exp.all_summaries()
    assert len(summaries) == 3


def test_all_summaries_ordered_vla_coarse_fine() -> None:
    results = [_make_result()]
    exp = _make_experiment(results, results, results)
    summaries = exp.all_summaries()
    assert summaries[0].condition == "vla_only"
    assert summaries[1].condition == "coarse_agentic"
    assert summaries[2].condition == "fine_agentic"


# ---------------------------------------------------------------------------
# ConditionSummary validation
# ---------------------------------------------------------------------------


def test_condition_summary_is_pydantic_model() -> None:
    s = ConditionSummary(
        condition="vla_only",
        n_episodes=3,
        n_completed=3,
        completion_rate=1.0,
        mean_subtasks_planned=1.0,
        mean_policy_calls=1.0,
        mean_retries=0.0,
        mean_replans=0.0,
    )
    assert s.condition == "vla_only"
    assert s.completion_rate == 1.0
