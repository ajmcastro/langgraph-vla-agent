"""Planning-granularity experiment (Milestone 6).

Compares three conditions on a shared set of mock scenarios:
  - vla_only      : full goal → single subtask (no decomposition)
  - coarse_agentic: 2-subtask decomposition via DeterministicPlanner
  - fine_agentic  : 5-subtask decomposition via DeterministicPlanner

All conditions run through the same AgentRunner/MockEnvironment path so
metric shapes are uniform and comparison is apples-to-apples.

Key limitation
--------------
MockEnvironment is deterministic: with a SUCCEED_AT_STEP scenario, all three
conditions complete at 100% regardless of planning granularity.  The
informative metrics in mock mode are COSTS (policy calls, subtask count),
not success rates.  Real performance differences require simulation or
hardware (M7+).
"""

from __future__ import annotations

import structlog
from pydantic import BaseModel, Field

from langgraph_vla_agent.agent.config import AgentConfig, Granularity
from langgraph_vla_agent.agent.state import AgentStatus
from langgraph_vla_agent.domain.context import EvaluationMode
from langgraph_vla_agent.environments.mock import MockScenario

_log = structlog.get_logger(__name__)

_EXPERIMENT_NOTE = (
    "Results are from mock evaluation mode. "
    "Completion rates, subtask counts, and policy-call counts reflect "
    "software behaviour under deterministic mock scenarios — "
    "not real closed-loop task performance. "
    "All three conditions will complete at the same rate when the mock "
    "environment is set to always succeed; the meaningful comparison is "
    "orchestration COST (policy calls, subtask overhead), not success rate."
)

# Ordered list of (Granularity enum, human-readable condition name)
_CONDITIONS: list[tuple[Granularity, str]] = [
    (Granularity.VLA_ONLY, "vla_only"),
    (Granularity.COARSE, "coarse_agentic"),
    (Granularity.FINE, "fine_agentic"),
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class EpisodeScenario(BaseModel):
    """One evaluation scenario: a goal text + mock environment settings."""

    goal: str
    mock_scenario: MockScenario = MockScenario.SUCCEED_AT_STEP
    succeed_at_step: int = Field(default=2, ge=1)


class ConditionResult(BaseModel):
    """Result of running one planning condition on one episode scenario."""

    condition: str
    episode_idx: int
    goal: str
    final_status: str
    n_subtasks_planned: int
    n_subtasks_completed: int
    n_subtasks_failed: int
    n_policy_calls: int  # = len(execution_history_references)
    retry_count: int
    replan_count: int

    @property
    def completed(self) -> bool:
        return self.final_status == AgentStatus.COMPLETED


class ConditionSummary(BaseModel):
    """Aggregate metrics for one condition across all episode scenarios."""

    condition: str
    n_episodes: int
    n_completed: int
    completion_rate: float
    mean_subtasks_planned: float
    mean_policy_calls: float
    mean_retries: float
    mean_replans: float


class GranularityExperimentResult(BaseModel):
    """Full planning-granularity experiment result: 3 conditions x N episodes."""

    results_by_condition: dict[str, list[ConditionResult]]
    n_episodes: int
    evaluation_note: str = _EXPERIMENT_NOTE

    def condition_summary(self, condition: str) -> ConditionSummary:
        """Compute aggregate metrics for one condition."""
        results = self.results_by_condition.get(condition, [])
        n = len(results)
        if n == 0:
            return ConditionSummary(
                condition=condition,
                n_episodes=0,
                n_completed=0,
                completion_rate=0.0,
                mean_subtasks_planned=0.0,
                mean_policy_calls=0.0,
                mean_retries=0.0,
                mean_replans=0.0,
            )
        n_completed = sum(1 for r in results if r.completed)
        return ConditionSummary(
            condition=condition,
            n_episodes=n,
            n_completed=n_completed,
            completion_rate=n_completed / n,
            mean_subtasks_planned=sum(r.n_subtasks_planned for r in results) / n,
            mean_policy_calls=sum(r.n_policy_calls for r in results) / n,
            mean_retries=sum(r.retry_count for r in results) / n,
            mean_replans=sum(r.replan_count for r in results) / n,
        )

    def all_summaries(self) -> list[ConditionSummary]:
        """Return ConditionSummary for every condition, in experiment order."""
        return [self.condition_summary(name) for _, name in _CONDITIONS]

    def summary_lines(self) -> list[str]:
        """Return a human-readable comparison table as a list of strings."""
        summaries = self.all_summaries()
        header = (
            f"{'Condition':<18} {'Episodes':>9} {'Completed':>10} "
            f"{'Rate%':>7} {'Subtasks':>9} {'PolCalls':>9} "
            f"{'Retries':>8} {'Replans':>8}"
        )
        sep = "-" * len(header)
        rows: list[str] = []
        for s in summaries:
            rows.append(
                f"{s.condition:<18} {s.n_episodes:>9} {s.n_completed:>10} "
                f"{s.completion_rate * 100:>6.1f}% {s.mean_subtasks_planned:>9.1f} "
                f"{s.mean_policy_calls:>9.1f} {s.mean_retries:>8.1f} "
                f"{s.mean_replans:>8.1f}"
            )
        return [header, sep, *rows, "", f"Note: {self.evaluation_note}"]


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def run_granularity_experiment(
    scenarios: list[EpisodeScenario],
    *,
    max_retries: int = 2,
    max_replans: int = 1,
) -> GranularityExperimentResult:
    """Run the 3-condition planning-granularity experiment.

    For each (condition, scenario) pair, builds a fresh AgentRunner wired with
    the appropriate planner and mock environment, runs one episode, and records
    the resulting AgentState metrics.

    Parameters
    ----------
    scenarios:
        List of evaluation scenarios (goal + MockScenario settings). Each
        scenario is run under all three conditions with identical settings.
    max_retries:
        Maximum retry attempts per subtask before replanning or failing.
    max_replans:
        Maximum replanning cycles per episode before failing.

    Returns
    -------
    GranularityExperimentResult
        Per-condition results and aggregate summaries.
    """
    # Import here to keep this module importable without the [agent] extra.
    from langgraph_vla_agent.agent.runner import make_mock_runner

    results_by_condition: dict[str, list[ConditionResult]] = {name: [] for _, name in _CONDITIONS}

    for granularity, condition_name in _CONDITIONS:
        config = AgentConfig(
            granularity=granularity,
            evaluation_mode=EvaluationMode.MOCK,
            max_retries=max_retries,
            max_replans=max_replans,
        )
        _log.info(
            "experiment.condition_start", condition=condition_name, n_scenarios=len(scenarios)
        )

        for i, scenario in enumerate(scenarios):
            runner = make_mock_runner(
                config=config,
                scenario=scenario.mock_scenario,
                succeed_at_step=scenario.succeed_at_step,
            )
            state = runner.run(scenario.goal)

            plan = state.get("plan")
            result = ConditionResult(
                condition=condition_name,
                episode_idx=i,
                goal=scenario.goal,
                final_status=str(state.get("final_status", "unknown")),
                n_subtasks_planned=len(plan.subtasks) if plan else 0,
                n_subtasks_completed=len(state.get("completed_subtask_ids", [])),
                n_subtasks_failed=len(state.get("failed_subtask_ids", [])),
                n_policy_calls=len(state.get("execution_history_references", [])),
                retry_count=state.get("retry_count", 0),
                replan_count=state.get("replan_count", 0),
            )
            results_by_condition[condition_name].append(result)
            _log.info(
                "experiment.episode_done",
                condition=condition_name,
                episode_idx=i,
                final_status=result.final_status,
                n_subtasks=result.n_subtasks_planned,
                n_policy_calls=result.n_policy_calls,
            )

    return GranularityExperimentResult(
        results_by_condition=results_by_condition,
        n_episodes=len(scenarios),
    )
