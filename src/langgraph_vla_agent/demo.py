"""Portfolio demo functions for LangGraph VLA Agent.

Three self-contained functions — one per evaluation mode — that can be called
independently by the CLI demo script or by unit tests.  Each returns a plain
dict so the caller can format and display results however it likes.

Evaluation modes
----------------
run_replay_demo()
    Offline/replay evaluation — OfflineEvaluator + fixture episodes +
    MockRobotPolicy.  No extras beyond core + dev.  Measures L1/L2 action
    prediction error against 3 fixture episodes.

run_mock_agent_demo()
    Mock LangGraph agent — DeterministicPlanner + MockRobotPolicy +
    MockEnvironment.  Requires the [agent] extra (make setup-agent).
    Proves graph routing, retry/replan paths, and safety gate.

run_simulation_demo()
    Closed-loop simulation experiment — hard scenario where vla_only FAILS
    and agentic conditions SUCCEED.  Requires the [agent] extra.
    Proves that per-subtask budget constraints produce differentiated outcomes.
"""

from __future__ import annotations

from pathlib import Path

import structlog

_log = structlog.get_logger(__name__)

# Project-root-relative path to the committed fixture episodes.
# demo.py lives at src/langgraph_vla_agent/demo.py → parents[2] = project root.
_FIXTURE_DIR = Path(__file__).parents[2] / "data" / "fixtures" / "episodes"

# Hard simulation scenario parameters (mirrors run_simulation.py --hard).
_SIM_TOTAL_PROGRESS = 0.5
_SIM_PROGRESS_PER_STEP = 0.15
_SIM_MAX_STEPS = 5
_SIM_SEED = 42

_SIM_GOALS = [
    "pick up the cube and place it in the bin",
    "pick up the red block and put it in the tray",
    "grasp the object and move it to the target zone",
]


def run_replay_demo() -> dict[str, object]:
    """Offline/replay evaluation on committed fixture episodes.

    Uses MockRobotPolicy (constant zero actions) so that L1 error is nonzero
    and interpretable as the gap between a naive baseline and the recorded
    ground-truth actions.

    Returns
    -------
    dict with keys:
        mode        -- "replay"
        n_episodes  -- number of fixture episodes evaluated
        n_steps     -- total steps across all episodes
        l1_mean     -- mean L1 action error across all steps
        l1_std      -- std of L1 error
        l2_mean     -- mean L2 action error across all steps
    """
    from langgraph_vla_agent.datasets.store import FixtureEpisodeStore
    from langgraph_vla_agent.evaluation.offline import OfflineEvaluator
    from langgraph_vla_agent.policies.mock import MockPolicyBehavior, MockRobotPolicy

    store = FixtureEpisodeStore(_FIXTURE_DIR)
    episode_ids = store.list_episodes()
    policy = MockRobotPolicy(behavior=MockPolicyBehavior.ALWAYS_VALID)
    evaluator = OfflineEvaluator(
        policy=policy,
        store=store,
        model_id="MockRobotPolicy(zeros)",
        dataset_id="fixture_episodes",
    )
    result = evaluator.evaluate(episode_ids, run_id="demo-replay")
    agg = result.aggregate

    _log.info(
        "demo.replay_done",
        n_episodes=result.n_episodes,
        l1_mean=round(agg.l1_mean, 4),
    )
    return {
        "mode": "replay",
        "n_episodes": result.n_episodes,
        "n_steps": agg.n_steps,
        "l1_mean": agg.l1_mean,
        "l1_std": agg.l1_std,
        "l2_mean": agg.l2_mean,
    }


def run_mock_agent_demo() -> dict[str, object]:
    """Mock LangGraph agent run on a pick-and-place goal.

    Requires the [agent] extra: uv sync --extra dev --extra agent

    Uses DeterministicPlanner (coarse, 2 subtasks) + MockRobotPolicy +
    MockEnvironment. No LLM key, GPU, or dataset needed.

    Returns
    -------
    dict with keys:
        mode                -- "mock"
        final_status        -- AgentStatus string (e.g. "completed")
        n_subtasks_planned  -- number of subtasks in the plan
        n_subtasks_completed-- subtasks that reached terminal success
        n_policy_calls      -- total policy.act() calls (= execution refs)
        retry_count         -- retries on the last subtask
        replan_count        -- full replanning cycles
    """
    from langgraph_vla_agent.agent.runner import make_mock_runner

    runner = make_mock_runner()
    goal = "pick up the cube and place it in the bin"
    state = runner.run(goal, run_id="demo-mock")

    plan = state.get("plan")
    result = {
        "mode": "mock",
        "final_status": str(state.get("final_status", "unknown")),
        "n_subtasks_planned": len(plan.subtasks) if plan else 0,
        "n_subtasks_completed": len(state.get("completed_subtask_ids", [])),
        "n_policy_calls": len(state.get("execution_history_references", [])),
        "retry_count": state.get("retry_count", 0),
        "replan_count": state.get("replan_count", 0),
    }
    _log.info(
        "demo.mock_agent_done",
        final_status=result["final_status"],
        n_subtasks=result["n_subtasks_planned"],
    )
    return result


def run_simulation_demo() -> dict[str, object]:
    """Hard-scenario closed-loop simulation experiment.

    Requires the [agent] extra: uv sync --extra dev --extra agent

    Runs the 3-condition experiment (vla_only / coarse_agentic / fine_agentic)
    in hard mode: total_progress=0.5, max_steps=5.  vla_only cannot reach its
    threshold in the step budget and FAILS; agentic conditions SUCCEED.

    Returns
    -------
    dict with keys:
        mode        -- "simulation"
        vla_rate    -- completion rate for vla_only (0.0 on hard scenario)
        coarse_rate -- completion rate for coarse_agentic (1.0 on hard scenario)
        fine_rate   -- completion rate for fine_agentic (1.0 on hard scenario)
        scenario    -- dict of scenario parameters
    """
    from langgraph_vla_agent.environments.simulation import SimulationScenario
    from langgraph_vla_agent.evaluation.simulation import (
        SimulationEpisodeScenario,
        run_simulation_experiment,
    )

    scenario = SimulationScenario(
        total_progress=_SIM_TOTAL_PROGRESS,
        progress_per_step=_SIM_PROGRESS_PER_STEP,
        max_steps_per_subtask=_SIM_MAX_STEPS,
        noise_scale=0.0,
        seed=_SIM_SEED,
    )
    episodes = [SimulationEpisodeScenario(goal=g, scenario=scenario) for g in _SIM_GOALS]
    exp = run_simulation_experiment(episodes, max_retries=0, max_replans=0)

    vla_rate = exp.condition_summary("vla_only").completion_rate
    coarse_rate = exp.condition_summary("coarse_agentic").completion_rate
    fine_rate = exp.condition_summary("fine_agentic").completion_rate

    _log.info(
        "demo.simulation_done",
        vla_rate=vla_rate,
        coarse_rate=coarse_rate,
        fine_rate=fine_rate,
    )
    return {
        "mode": "simulation",
        "vla_rate": vla_rate,
        "coarse_rate": coarse_rate,
        "fine_rate": fine_rate,
        "scenario": {
            "total_progress": scenario.total_progress,
            "progress_per_step": scenario.progress_per_step,
            "max_steps_per_subtask": scenario.max_steps_per_subtask,
        },
    }
