"""Simulation-mode planning-granularity experiment (Milestone 7).

Extends the M6 mock experiment by substituting SimulationEnvironment for
MockEnvironment.  The key difference: actions now affect world state and
the success predicate is closed-loop.

What this adds over M6
----------------------
In mock mode all three conditions complete at 100% on success scenarios
because the environment is scripted.  In simulation mode, the per-subtask
action budget is fixed and the success threshold is split proportionally
across subtasks.  Under a *constrained* scenario this means:

    - vla_only:       1 subtask, threshold=total_progress       → may FAIL
    - coarse_agentic: 2 subtasks, threshold=total_progress/2    → easier
    - fine_agentic:   5 subtasks, threshold=total_progress/5    → easiest

With MockRobotPolicy (zero actions), progress per step = 0.5 * progress_per_step.
The constrained scenario is calibrated so that vla_only cannot reach its
threshold within max_steps_per_subtask but coarse/fine agentic can.

Key limitation
--------------
This is still a *toy* simulation with a scalar progress model.  It is NOT
a physics simulator (no MuJoCo, no rigid-body dynamics, no camera rendering).
Results prove the *software architecture* supports closed-loop evaluation and
that decomposition *can* improve task completion when per-subtask budgets are
limited.  They do not prove real robot performance.  Cite the evaluation mode
on every result: EvaluationMode.SIMULATION.
"""

from __future__ import annotations

import structlog
from pydantic import BaseModel, Field

from langgraph_vla_agent.agent.config import AgentConfig, Granularity
from langgraph_vla_agent.agent.state import AgentStatus
from langgraph_vla_agent.domain.context import EvaluationMode
from langgraph_vla_agent.environments.simulation import SimulationScenario

_log = structlog.get_logger(__name__)

_SIMULATION_NOTE = (
    "Results are from toy simulation mode (scalar progress model, no physics engine). "
    "Actions affect world state and outcomes are closed-loop — unlike mock/replay mode "
    "where actions are ignored.  However, this is not a real physics simulator "
    "(no MuJoCo, no rigid-body dynamics, no camera rendering).  Results prove that "
    "the software architecture supports closed-loop evaluation and that finer subtask "
    "decomposition can improve completion when per-subtask action budgets are limited. "
    "They do not predict real-robot performance."
)

# Maps each condition to its fixed subtask count.
# Must stay consistent with DeterministicPlanner / VlaOnlyPlanner definitions.
_CONDITION_N_SUBTASKS: dict[str, int] = {
    "vla_only": 1,
    "coarse_agentic": 2,
    "fine_agentic": 5,
}

# Ordered list of (Granularity enum, human-readable condition name)
_CONDITIONS: list[tuple[Granularity, str]] = [
    (Granularity.VLA_ONLY, "vla_only"),
    (Granularity.COARSE, "coarse_agentic"),
    (Granularity.FINE, "fine_agentic"),
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class SimulationEpisodeScenario(BaseModel):
    """One evaluation scenario for the simulation experiment."""

    goal: str
    scenario: SimulationScenario = Field(default_factory=SimulationScenario)


class SimulationConditionResult(BaseModel):
    """Result of running one planning condition on one simulation episode."""

    condition: str
    episode_idx: int
    goal: str
    final_status: str
    n_subtasks_planned: int
    n_subtasks_completed: int
    n_subtasks_failed: int
    n_policy_calls: int
    retry_count: int
    replan_count: int
    per_subtask_threshold: float
    total_progress_required: float

    @property
    def completed(self) -> bool:
        return self.final_status == AgentStatus.COMPLETED


class SimulationConditionSummary(BaseModel):
    """Aggregate metrics for one condition across all simulation episodes."""

    condition: str
    n_episodes: int
    n_completed: int
    completion_rate: float
    mean_subtasks_planned: float
    mean_policy_calls: float
    mean_retries: float
    mean_replans: float
    per_subtask_threshold: float
    total_progress_required: float


class SimulationExperimentResult(BaseModel):
    """Full simulation-mode planning-granularity experiment result."""

    results_by_condition: dict[str, list[SimulationConditionResult]]
    n_episodes: int
    scenario: SimulationScenario
    evaluation_note: str = _SIMULATION_NOTE

    def condition_summary(self, condition: str) -> SimulationConditionSummary:
        results = self.results_by_condition.get(condition, [])
        n = len(results)
        n_subtasks = _CONDITION_N_SUBTASKS.get(condition, 1)
        threshold = self.scenario.total_progress / n_subtasks
        if n == 0:
            return SimulationConditionSummary(
                condition=condition,
                n_episodes=0,
                n_completed=0,
                completion_rate=0.0,
                mean_subtasks_planned=0.0,
                mean_policy_calls=0.0,
                mean_retries=0.0,
                mean_replans=0.0,
                per_subtask_threshold=threshold,
                total_progress_required=self.scenario.total_progress,
            )
        n_completed = sum(1 for r in results if r.completed)
        return SimulationConditionSummary(
            condition=condition,
            n_episodes=n,
            n_completed=n_completed,
            completion_rate=n_completed / n,
            mean_subtasks_planned=sum(r.n_subtasks_planned for r in results) / n,
            mean_policy_calls=sum(r.n_policy_calls for r in results) / n,
            mean_retries=sum(r.retry_count for r in results) / n,
            mean_replans=sum(r.replan_count for r in results) / n,
            per_subtask_threshold=threshold,
            total_progress_required=self.scenario.total_progress,
        )

    def all_summaries(self) -> list[SimulationConditionSummary]:
        return [self.condition_summary(name) for _, name in _CONDITIONS]

    def summary_lines(self) -> list[str]:
        summaries = self.all_summaries()
        header = (
            f"{'Condition':<18} {'Episodes':>9} {'Completed':>10} "
            f"{'Rate%':>7} {'Threshold':>10} {'Subtasks':>9} "
            f"{'PolCalls':>9} {'Retries':>8} {'Replans':>8}"
        )
        sep = "-" * len(header)
        rows: list[str] = []
        for s in summaries:
            rows.append(
                f"{s.condition:<18} {s.n_episodes:>9} {s.n_completed:>10} "
                f"{s.completion_rate * 100:>6.1f}% "
                f"{s.per_subtask_threshold:>10.3f} "
                f"{s.mean_subtasks_planned:>9.1f} "
                f"{s.mean_policy_calls:>9.1f} "
                f"{s.mean_retries:>8.1f} "
                f"{s.mean_replans:>8.1f}"
            )
        return [header, sep, *rows, "", f"Note: {self.evaluation_note}"]


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def run_simulation_experiment(
    scenarios: list[SimulationEpisodeScenario],
    *,
    max_retries: int = 0,
    max_replans: int = 0,
) -> SimulationExperimentResult:
    """Run the 3-condition planning-granularity experiment in simulation mode.

    For each (condition, scenario) pair, builds a fresh AgentRunner wired with
    a SimulationEnvironment whose success_threshold is scaled by the number of
    subtasks the condition produces.  This keeps total task difficulty constant
    across conditions while making per-subtask targets proportionally easier.

    Parameters
    ----------
    scenarios:
        List of simulation scenarios.  All episodes use the same
        SimulationScenario parameters (from scenarios[0]) for threshold
        calculation; individual goal text varies.
    max_retries:
        Maximum retry attempts per subtask.  Defaults to 0 so failures are
        immediate — this isolates the effect of decomposition from recovery.
    max_replans:
        Maximum replanning cycles per episode.

    Returns
    -------
    SimulationExperimentResult
        Per-condition results and aggregate summaries.

    Notes
    -----
    Requires the [agent] extra (LangGraph).  This import is deferred inside
    the function body so simulation.py is importable without the extra, keeping
    unit tests runnable under ``make check``.
    """
    # Import here to keep this module importable without the [agent] extra.
    from langgraph_vla_agent.agent.runner import make_simulation_runner

    base_scenario = scenarios[0].scenario if scenarios else SimulationScenario()

    results_by_condition: dict[str, list[SimulationConditionResult]] = {
        name: [] for _, name in _CONDITIONS
    }

    for granularity, condition_name in _CONDITIONS:
        n_subtasks = _CONDITION_N_SUBTASKS[condition_name]
        per_subtask_threshold = base_scenario.total_progress / n_subtasks

        config = AgentConfig(
            granularity=granularity,
            evaluation_mode=EvaluationMode.SIMULATION,
            max_retries=max_retries,
            max_replans=max_replans,
        )

        _log.info(
            "sim_experiment.condition_start",
            condition=condition_name,
            n_subtasks=n_subtasks,
            per_subtask_threshold=round(per_subtask_threshold, 4),
            n_scenarios=len(scenarios),
        )

        for i, ep_scenario in enumerate(scenarios):
            runner = make_simulation_runner(
                config=config,
                scenario=ep_scenario.scenario,
                n_subtasks=n_subtasks,
            )
            state = runner.run(ep_scenario.goal)

            plan = state.get("plan")
            result = SimulationConditionResult(
                condition=condition_name,
                episode_idx=i,
                goal=ep_scenario.goal,
                final_status=str(state.get("final_status", "unknown")),
                n_subtasks_planned=len(plan.subtasks) if plan else 0,
                n_subtasks_completed=len(state.get("completed_subtask_ids", [])),
                n_subtasks_failed=len(state.get("failed_subtask_ids", [])),
                n_policy_calls=len(state.get("execution_history_references", [])),
                retry_count=state.get("retry_count", 0),
                replan_count=state.get("replan_count", 0),
                per_subtask_threshold=per_subtask_threshold,
                total_progress_required=base_scenario.total_progress,
            )
            results_by_condition[condition_name].append(result)
            _log.info(
                "sim_experiment.episode_done",
                condition=condition_name,
                episode_idx=i,
                final_status=result.final_status,
                n_subtasks=result.n_subtasks_planned,
                threshold=round(per_subtask_threshold, 4),
            )

    return SimulationExperimentResult(
        results_by_condition=results_by_condition,
        n_episodes=len(scenarios),
        scenario=base_scenario,
    )
