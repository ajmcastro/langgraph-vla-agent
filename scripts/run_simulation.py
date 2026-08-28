"""M7 simulation-mode planning-granularity experiment runner.

Compares three planning conditions using SimulationEnvironment — a closed-loop
toy physics environment where actions affect world state and outcomes:

  - vla_only      : full goal as single subtask; threshold = total_progress / 1
  - coarse_agentic: 2-subtask decomposition;     threshold = total_progress / 2
  - fine_agentic  : 5-subtask decomposition;     threshold = total_progress / 5

Key difference from mock mode (M6): actions are NOT ignored.  The policy's
output determines how fast progress is made.  With a constrained per-subtask
budget, finer decomposition can succeed where VLA-only fails.

Usage
-----
    uv run python scripts/run_simulation.py               # easy mode (all succeed)
    uv run python scripts/run_simulation.py --hard        # constrained (VLA-only fails)
    uv run python scripts/run_simulation.py --progress 0.4 --steps 6
    uv run python scripts/run_simulation.py --hard --quiet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langgraph_vla_agent.environments.simulation import SimulationScenario
from langgraph_vla_agent.evaluation.simulation import (
    SimulationEpisodeScenario,
    run_simulation_experiment,
)

# ---------------------------------------------------------------------------
# Scenario parameters
# ---------------------------------------------------------------------------

# Easy: all conditions succeed with MockRobotPolicy (zero actions).
# vla_only threshold = 0.3, steps needed ≈ 4, budget = 10 → succeeds.
_EASY = SimulationScenario(
    total_progress=0.3,
    progress_per_step=0.15,
    max_steps_per_subtask=10,
    noise_scale=0.0,
    seed=42,
)

# Hard: vla_only threshold = 0.5, steps needed ≈ 7, budget = 5 → FAILS.
#       coarse threshold = 0.25, steps needed ≈ 4 → SUCCEEDS.
#       fine threshold   = 0.10, steps needed ≈ 2 → SUCCEEDS.
_HARD = SimulationScenario(
    total_progress=0.5,
    progress_per_step=0.15,
    max_steps_per_subtask=5,
    noise_scale=0.0,
    seed=42,
)

_GOALS = [
    "pick up the cube and place it in the bin",
    "pick up the red block and put it in the tray",
    "grasp the object and move it to the target zone",
]

# ---------------------------------------------------------------------------
# Explanatory text
# ---------------------------------------------------------------------------

_HEADER = """
╔══════════════════════════════════════════════════════════════════════╗
║  M7 Simulation-Mode Planning-Granularity Experiment                  ║
║  Evaluation mode: SIMULATION (closed-loop toy physics)               ║
╚══════════════════════════════════════════════════════════════════════╝
"""

_WHAT_THIS_PROVES = """
What this PROVES (simulation evaluation):
  • SimulationEnvironment is closed-loop: the policy's actions affect progress.
  • In the EASY mode (low threshold, generous budget), all three conditions
    complete at 100% — consistent with mock mode.
  • In the HARD mode (high threshold, tight budget):
      - vla_only FAILS: one subtask must reach threshold=0.5 in 5 steps,
        but MockRobotPolicy (zero actions) only achieves 0.375 in 5 steps.
      - coarse_agentic SUCCEEDS: each subtask threshold=0.25, achievable in 4 steps.
      - fine_agentic SUCCEEDS: each subtask threshold=0.10, achievable in 2 steps.
  • Finer decomposition CAN improve task completion when per-subtask budgets are
    limited — a result mock mode cannot produce.
"""

_WHAT_THIS_DOES_NOT_PROVE = """
What this does NOT prove:
  • Real robot performance. SimulationEnvironment is a toy scalar progress model —
    no MuJoCo, no rigid-body dynamics, no camera rendering, no real physics.
  • That SmolVLA or any real VLA policy benefits from decomposition in the same way.
    MockRobotPolicy always returns zero actions, so the only variable is the
    per-subtask success threshold, not policy quality.
  • That the calibration (total_progress=0.5, progress_per_step=0.15, max_steps=5)
    generalises to real tasks. The numbers are chosen to illustrate the mechanism,
    not to model a real manipulation scenario.
  • Sim-to-real transfer. Connecting the graph to a real simulator (MuJoCo via
    gym-pusht or gym-aloha) or to hardware is future work.
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M7 simulation-mode granularity experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--hard",
        action="store_true",
        help="Use constrained scenario where VLA-only fails (default: easy)",
    )
    parser.add_argument(
        "--progress",
        type=float,
        default=None,
        help="Override total_progress (default: 0.3 easy / 0.5 hard)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Override max_steps_per_subtask (default: 10 easy / 5 hard)",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.0,
        help="Gaussian noise scale on progress (default: 0.0, deterministic)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Skip explanatory text and print only the results table",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    base = _HARD if args.hard else _EASY
    scenario = SimulationScenario(
        total_progress=args.progress if args.progress is not None else base.total_progress,
        progress_per_step=base.progress_per_step,
        max_steps_per_subtask=args.steps if args.steps is not None else base.max_steps_per_subtask,
        noise_scale=args.noise,
        seed=base.seed,
    )

    episodes = [SimulationEpisodeScenario(goal=g, scenario=scenario) for g in _GOALS]

    if not args.quiet:
        print(_HEADER)
        mode = "HARD (constrained)" if args.hard else "EASY (generous budget)"
        print(f"Mode: {mode}")
        print(f"Scenarios: {len(episodes)}  |  Conditions: 3")
        print(f"total_progress={scenario.total_progress}  "
              f"progress_per_step={scenario.progress_per_step}  "
              f"max_steps_per_subtask={scenario.max_steps_per_subtask}")
        print(f"Per-subtask thresholds: "
              f"vla_only={scenario.total_progress:.3f}  "
              f"coarse={scenario.total_progress/2:.3f}  "
              f"fine={scenario.total_progress/5:.3f}")
        print()

    result = run_simulation_experiment(episodes, max_retries=0, max_replans=0)

    print()
    print("── Experiment Results ──────────────────────────────────────────────")
    for line in result.summary_lines():
        print(line)

    if not args.quiet:
        print(_WHAT_THIS_PROVES)
        print(_WHAT_THIS_DOES_NOT_PROVE)


if __name__ == "__main__":
    main()
