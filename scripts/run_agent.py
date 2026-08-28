"""Run the LangGraph VLA agent in mock or replay mode.

Usage
-----
Mock mode (no LLM, no GPU — works with just [agent] extra):
    make evaluate-agent
    uv run python scripts/run_agent.py
    uv run python scripts/run_agent.py --granularity fine --goal "pick up the cube"

Deterministic planner, custom goal:
    uv run python scripts/run_agent.py \\
        --granularity coarse \\
        --goal "pick up the cube and place it in the bin" \\
        --max-retries 2 \\
        --max-replans 1

What this proves (mock mode)
-----------------------------
- Goal decomposition into subtasks (DeterministicPlanner, no LLM)
- LangGraph graph routing: understand → plan → select → safety → execute → verify
- Bounded retry and replan logic
- Safety gate (allowlist check before each subtask)
- Structured traces via structlog

What this does NOT prove
------------------------
- Real sensorimotor capability (MockRobotPolicy outputs constant zeros)
- Closed-loop task success (MockEnvironment uses scripted termination)
- LLM planning quality (LLMTaskPlanner requires an API key and --planner llm)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import structlog

from langgraph_vla_agent.agent.config import AgentConfig, Granularity, PlannerType
from langgraph_vla_agent.agent.runner import make_mock_runner
from langgraph_vla_agent.domain.context import EvaluationMode
from langgraph_vla_agent.environments.mock import MockScenario

_log = structlog.get_logger(__name__)

_DEFAULT_GOAL = "pick up the cube and place it in the bin"


def _print_result(state: dict) -> None:  # type: ignore[type-arg]
    print()
    print("=" * 60)
    print("  Agent run complete")
    print("=" * 60)
    print(f"  run_id              : {state.get('run_id', '?')}")
    print(f"  goal                : {state.get('goal', {}).text if state.get('goal') else '?'}")  # type: ignore[union-attr]
    print(f"  final_status        : {state.get('final_status', '?')}")
    print(f"  completed subtasks  : {len(state.get('completed_subtask_ids', []))}")
    print(f"  failed subtasks     : {len(state.get('failed_subtask_ids', []))}")
    print(f"  retry_count         : {state.get('retry_count', 0)}")
    print(f"  replan_count        : {state.get('replan_count', 0)}")
    print(f"  execution refs      : {len(state.get('execution_history_references', []))}")
    if state.get("error_message"):
        print(f"  error_message       : {state['error_message']}")
    if state.get("safety_rejection_reason"):
        print(f"  safety_rejection    : {state['safety_rejection_reason']}")
    print()
    print("  What these numbers mean:")
    print("  - All execution is mock (scripted environment, zero-action policy).")
    print("  - 'completed subtasks' proves graph routing, not robot capability.")
    print("  - Evaluation mode: MOCK. No sensorimotor claims can be made.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the LangGraph VLA agent (mock mode by default).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--goal",
        default=_DEFAULT_GOAL,
        help="Natural-language goal for the agent (default: pick-and-place).",
    )
    parser.add_argument(
        "--granularity",
        choices=["coarse", "fine"],
        default="coarse",
        help="Subtask decomposition granularity (default: coarse).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Maximum retry attempts per subtask (default: 2).",
    )
    parser.add_argument(
        "--max-replans",
        type=int,
        default=1,
        help="Maximum replanning cycles per episode (default: 1).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional correlation ID for the run (auto-generated if not set).",
    )
    args = parser.parse_args()

    granularity = Granularity.COARSE if args.granularity == "coarse" else Granularity.FINE

    config = AgentConfig(
        max_retries=args.max_retries,
        max_replans=args.max_replans,
        planner_type=PlannerType.DETERMINISTIC,
        granularity=granularity,
        evaluation_mode=EvaluationMode.MOCK,
    )

    runner = make_mock_runner(config=config, scenario=MockScenario.SUCCEED_AT_STEP, succeed_at_step=2)

    print()
    print(f"  Goal       : {args.goal}")
    print(f"  Granularity: {args.granularity}")
    print(f"  Max retries: {args.max_retries}  Max replans: {args.max_replans}")
    print(f"  Mode       : MOCK (DeterministicPlanner + MockPolicy + MockEnvironment)")
    print()

    state = runner.run(args.goal, run_id=args.run_id)
    _print_result(state)


if __name__ == "__main__":
    main()
