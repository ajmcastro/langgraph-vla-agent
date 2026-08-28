"""Portfolio demo for LangGraph VLA Agent (Milestone 8).

Runs three evaluation modes in sequence to showcase the full system:

  1. Offline / replay — OfflineEvaluator + fixture episodes + MockRobotPolicy
     No [agent] extra needed.  Shows evaluation infrastructure and L1 metrics.

  2. Mock agent      — LangGraph + DeterministicPlanner + MockRobotPolicy
     Requires [agent] extra (make setup-agent).
     Shows graph routing, subtask decomposition, safety gate.

  3. Simulation      — SimulationEnvironment + hard 3-condition experiment
     Requires [agent] extra (make setup-agent).
     Shows closed-loop differentiation: vla_only FAILS, agentic SUCCEEDS.

All modes are deterministic. No LLM key, GPU, dataset, or robot required.

Usage
-----
    make run-demo                          # all three modes
    uv run python scripts/run_demo.py
    uv run python scripts/run_demo.py --mode replay
    uv run python scripts/run_demo.py --mode mock
    uv run python scripts/run_demo.py --mode simulation
    uv run python scripts/run_demo.py --quiet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from langgraph_vla_agent.demo import run_replay_demo

_HEADER = """
╔══════════════════════════════════════════════════════════════════════╗
║  LangGraph VLA Agent — Portfolio Demo (M8)                           ║
║  All modes: deterministic, no GPU, no LLM key, no robot required     ║
╚══════════════════════════════════════════════════════════════════════╝
"""

_AGENT_EXTRA_MSG = (
    "\n  [agent] extra not installed — skipping this mode.\n"
    "  Run: make setup-agent   then retry.\n"
)


# ---------------------------------------------------------------------------
# Section printers
# ---------------------------------------------------------------------------


def _section(title: str, quiet: bool) -> None:
    if not quiet:
        print()
        print(f"{'─' * 70}")
        print(f"  {title}")
        print(f"{'─' * 70}")


def _run_replay(quiet: bool) -> None:
    _section("MODE 1 — Offline/Replay Evaluation (no [agent] needed)", quiet)
    result = run_replay_demo()
    print(f"  Episodes evaluated : {result['n_episodes']}")
    print(f"  Steps evaluated    : {result['n_steps']}")
    print(f"  L1 error (mean)    : {result['l1_mean']:.4f}  ±  {result['l1_std']:.4f}")
    print(f"  L2 error (mean)    : {result['l2_mean']:.4f}")
    if not quiet:
        print()
        print("  Interpretation: MockRobotPolicy returns zero actions; L1 error measures")
        print("  the gap between a naive zero baseline and recorded ground-truth actions.")
        print("  This proves the evaluation infrastructure works — not that any policy")
        print("  is good. Connecting SmolVLA would give a meaningful non-zero number.")


def _run_mock(quiet: bool) -> None:
    _section("MODE 2 — Mock LangGraph Agent Run (requires [agent] extra)", quiet)
    try:
        from langgraph_vla_agent.demo import run_mock_agent_demo
    except ImportError:
        print(_AGENT_EXTRA_MSG)
        return
    try:
        result = run_mock_agent_demo()
    except Exception as exc:
        print(f"  Error: {exc}")
        return
    print(f"  Goal               : pick up the cube and place it in the bin")
    print(f"  Planning           : DeterministicPlanner (coarse, 2 subtasks)")
    print(f"  Final status       : {result['final_status']}")
    print(f"  Subtasks planned   : {result['n_subtasks_planned']}")
    print(f"  Subtasks completed : {result['n_subtasks_completed']}")
    print(f"  Policy calls       : {result['n_policy_calls']}")
    print(f"  Retries / Replans  : {result['retry_count']} / {result['replan_count']}")
    if not quiet:
        print()
        print("  Interpretation: the full LangGraph graph ran — understand_goal →")
        print("  create_plan → select → safety_check → execute → verify → COMPLETED.")
        print("  MockRobotPolicy returns zeros; success is scripted (MockEnvironment).")
        print("  This proves software behavior (routing, safety, retries) — not robot skill.")


def _run_simulation(quiet: bool) -> None:
    _section(
        "MODE 3 — Simulation Experiment, Hard Scenario (requires [agent] extra)", quiet
    )
    try:
        from langgraph_vla_agent.demo import run_simulation_demo
    except ImportError:
        print(_AGENT_EXTRA_MSG)
        return
    try:
        result = run_simulation_demo()
    except Exception as exc:
        print(f"  Error: {exc}")
        return
    sc = result["scenario"]
    print(f"  total_progress={sc['total_progress']}  "
          f"progress_per_step={sc['progress_per_step']}  "
          f"max_steps={sc['max_steps_per_subtask']}")
    print()
    print(f"  {'Condition':<20} {'Threshold':>10} {'Rate':>8}")
    print(f"  {'─'*40}")
    vla_thresh = sc["total_progress"] / 1
    coarse_thresh = sc["total_progress"] / 2
    fine_thresh = sc["total_progress"] / 5
    print(f"  {'vla_only':<20} {vla_thresh:>10.3f} {result['vla_rate']:>7.0%}")
    print(f"  {'coarse_agentic':<20} {coarse_thresh:>10.3f} {result['coarse_rate']:>7.0%}")
    print(f"  {'fine_agentic':<20} {fine_thresh:>10.3f} {result['fine_rate']:>7.0%}")
    if not quiet:
        print()
        print("  Interpretation: vla_only must reach threshold=0.50 in 5 steps but only")
        print("  achieves 0.375 → FAILS. Coarse (threshold=0.25, needs 4 steps) and fine")
        print("  (threshold=0.10, needs 2 steps) succeed within their budgets.")
        print("  This differentiation is impossible in mock mode — it requires closed-loop")
        print("  evaluation where actions affect world state. SimulationEnvironment is a")
        print("  toy scalar model, not a physics simulator.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LangGraph VLA Agent portfolio demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["all", "replay", "mock", "simulation"],
        default="all",
        help="Which demo section to run (default: all)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only numbers, skip explanatory text",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    mode = args.mode
    quiet = args.quiet

    if not quiet:
        print(_HEADER)

    if mode in ("all", "replay"):
        _run_replay(quiet)

    if mode in ("all", "mock"):
        _run_mock(quiet)

    if mode in ("all", "simulation"):
        _run_simulation(quiet)

    if not quiet:
        print()
        print("  Demo complete. See docs/experiments.md for full experiment logs.")
        print("  See docs/evaluation.md for what each result proves and does not prove.")
        print()


if __name__ == "__main__":
    main()
