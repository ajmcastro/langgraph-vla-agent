"""M6 planning-granularity experiment runner.

Compares three planning conditions on a shared set of mock scenarios:
  - vla_only      : full goal passed as a single subtask — no decomposition
  - coarse_agentic: 2-subtask decomposition via DeterministicPlanner
  - fine_agentic  : 5-subtask decomposition via DeterministicPlanner

All conditions run through the same LangGraph agent and MockEnvironment.
No LLM key, no GPU, and no physical robot are required.

Usage
-----
    uv run python scripts/run_experiment.py
    uv run python scripts/run_experiment.py --max-retries 2 --max-replans 1
    uv run python scripts/run_experiment.py --fail-scenario
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langgraph_vla_agent.environments.mock import MockScenario
from langgraph_vla_agent.evaluation.experiment import EpisodeScenario, run_granularity_experiment

# ---------------------------------------------------------------------------
# Default scenario set
# ---------------------------------------------------------------------------

_SUCCESS_SCENARIOS: list[EpisodeScenario] = [
    EpisodeScenario(goal="pick up the cube and place it in the bin"),
    EpisodeScenario(goal="pick up the red block and put it in the tray"),
    EpisodeScenario(goal="grasp the object and move it to the target zone"),
    EpisodeScenario(goal="take the object and set it down at the goal position"),
    EpisodeScenario(goal="pick up the cube", succeed_at_step=3),
]

_FAIL_SCENARIO: EpisodeScenario = EpisodeScenario(
    goal="pick up the cube and place it in the bin",
    mock_scenario=MockScenario.FAIL_AT_STEP,
    succeed_at_step=999,
)

# ---------------------------------------------------------------------------
# Explanatory text
# ---------------------------------------------------------------------------

_HEADER = """
╔══════════════════════════════════════════════════════════════════════╗
║  M6 Planning-Granularity Experiment                                  ║
║  Evaluation mode: MOCK (deterministic, no GPU, no robot)             ║
╚══════════════════════════════════════════════════════════════════════╝
"""

_WHAT_THIS_PROVES = """
What this PROVES (mock evaluation):
  • VLA-only incurs the lowest orchestration cost: 1 policy call per episode.
  • Coarse agentic decomposes into 2 subtasks → 2 policy calls.
  • Fine agentic decomposes into 5 subtasks → 5 policy calls.
  • In mock mode all three conditions complete at 100% on success scenarios.
  • The retry/replan paths work correctly under the failure scenario.
  • All metric shapes (subtask counts, policy calls) are uniform across conditions.
"""

_WHAT_THIS_DOES_NOT_PROVE = """
What this does NOT prove:
  • That any planning condition improves task success in the real world.
    MockEnvironment is deterministic — it succeeds regardless of what the
    policy predicts. Real performance differences require simulation (M7)
    or hardware experiments.
  • That coarse or fine decomposition improves action quality.
    The policy called is MockRobotPolicy (returns constant valid actions),
    not SmolVLA. Connecting the graph to SmolVLA in offline/replay mode
    is the next step (M7 or an extension of M6).
  • That the subtask vocabulary matches SmolVLA's training distribution.
    The DeterministicPlanner subtask instructions are plausible but
    not validated against the svla_so100_pickplace dataset language.
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M6 planning-granularity experiment (mock mode)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Max retry attempts per subtask (default: 2)",
    )
    parser.add_argument(
        "--max-replans",
        type=int,
        default=1,
        help="Max replanning cycles per episode (default: 1)",
    )
    parser.add_argument(
        "--fail-scenario",
        action="store_true",
        help="Add a failure scenario to see retry/replan behaviour",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Skip explanatory text and print only the results table",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    scenarios = list(_SUCCESS_SCENARIOS)
    if args.fail_scenario:
        scenarios.append(_FAIL_SCENARIO)

    if not args.quiet:
        print(_HEADER)
        print(f"Scenarios: {len(scenarios)}  |  Conditions: 3")
        print(f"max_retries={args.max_retries}  max_replans={args.max_replans}")
        print()

    result = run_granularity_experiment(
        scenarios,
        max_retries=args.max_retries,
        max_replans=args.max_replans,
    )

    print()
    print("── Experiment Results ──────────────────────────────────────────────")
    for line in result.summary_lines():
        print(line)

    if not args.quiet:
        print(_WHAT_THIS_PROVES)
        print(_WHAT_THIS_DOES_NOT_PROVE)


if __name__ == "__main__":
    main()
