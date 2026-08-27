"""Offline policy evaluation script.

Usage
-----
Mock evaluation (no VLA extra needed):
    uv run python scripts/evaluate_policy.py --mode mock

Replay evaluation on fixture episodes (no VLA extra needed):
    uv run python scripts/evaluate_policy.py --mode replay

VLA evaluation (requires [vla] extra and model download):
    uv run python scripts/evaluate_policy.py --mode vla

Or via Makefile:
    make evaluate-mock
    make evaluate-replay
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np

from langgraph_vla_agent.datasets.episode import ReplayEpisode, ReplayStep
from langgraph_vla_agent.datasets.store import FixtureEpisodeStore
from langgraph_vla_agent.domain.context import EvaluationMode
from langgraph_vla_agent.evaluation import OfflineEvaluator
from langgraph_vla_agent.policies import MockRobotPolicy, ReplayRobotPolicy, vla_available
from langgraph_vla_agent.policies.mock import MockPolicyBehavior

_FIXTURE_DIR = Path(__file__).parents[1] / "data" / "fixtures" / "episodes"


def _make_demo_episode() -> ReplayEpisode:
    """Single synthetic episode for mock/demo mode."""
    steps = [
        ReplayStep(
            timestep=i,
            observation={"state": [0.0] * 6},
            action=[float(i) * 0.1] + [0.0] * 5,
            terminated=(i == 4),
            success=(i == 4),
        )
        for i in range(5)
    ]
    return ReplayEpisode(
        episode_id="demo-ep-001",
        instruction="pick up the block and place it in the bin",
        dataset_id="synthetic",
        action_dim=6,
        state_dim=6,
        steps=steps,
    )


class _SingleEpisodeStore:
    """Wraps a single episode as an EpisodeStore."""

    def __init__(self, episode: ReplayEpisode) -> None:
        self._ep = episode

    def list_episodes(self) -> list[str]:
        return [self._ep.episode_id]

    def load_episode(self, episode_id: str) -> ReplayEpisode:
        if episode_id != self._ep.episode_id:
            raise FileNotFoundError(episode_id)
        return self._ep


def _run_mock(args: argparse.Namespace) -> None:
    print("=== Mock evaluation (MockRobotPolicy / ALWAYS_VALID) ===")
    ep = _make_demo_episode()
    store = _SingleEpisodeStore(ep)
    policy = MockRobotPolicy(behavior=MockPolicyBehavior.ALWAYS_VALID)
    evaluator = OfflineEvaluator(
        policy,
        store,
        model_id="mock-always-valid",
        dataset_id="synthetic",
        evaluation_mode=EvaluationMode.MOCK,
    )
    result = evaluator.evaluate([ep.episode_id])
    _print_result(result)


def _run_replay(args: argparse.Namespace) -> None:
    if not _FIXTURE_DIR.exists():
        print(f"ERROR: Fixture directory not found: {_FIXTURE_DIR}")
        print("Run the project setup first — fixture JSON files must exist under data/fixtures/.")
        sys.exit(1)

    print("=== Replay evaluation (ReplayRobotPolicy / fixture episodes) ===")
    store = FixtureEpisodeStore(_FIXTURE_DIR)
    ep_ids = store.list_episodes()
    print(f"Found {len(ep_ids)} fixture episode(s): {ep_ids}")

    ep = store.load_episode(ep_ids[0])
    policy = ReplayRobotPolicy(ep)

    class _PerEpisodeReplayPolicy:
        def reset(self, context):  # type: ignore[override]
            ep_local = store.load_episode(context.episode_id)
            self._inner = ReplayRobotPolicy(ep_local)
            self._inner.reset(context)

        def act(self, obs, instruction):  # type: ignore[override]
            return self._inner.act(obs, instruction)

    evaluator = OfflineEvaluator(
        _PerEpisodeReplayPolicy(),  # type: ignore[arg-type]
        store,
        model_id="replay-baseline",
        dataset_id="fixture",
        evaluation_mode=EvaluationMode.REPLAY,
    )
    result = evaluator.evaluate(ep_ids)
    _print_result(result)


def _run_vla(args: argparse.Namespace) -> None:
    if not vla_available():
        print("ERROR: [vla] extra is not installed.")
        print("Install it with:  uv sync --extra dev --extra vla")
        sys.exit(1)

    if not _FIXTURE_DIR.exists():
        print(f"ERROR: Fixture directory not found: {_FIXTURE_DIR}")
        sys.exit(1)

    from langgraph_vla_agent.policies import SmolVLAPolicyAdapter

    model_id = getattr(args, "model_id", "lerobot/smolvla_base")
    print(f"=== VLA evaluation ({model_id}) ===")
    print("NOTE: First run downloads the model checkpoint from HuggingFace Hub (~500 MB).")

    store = FixtureEpisodeStore(_FIXTURE_DIR)
    ep_ids = store.list_episodes()
    print(f"Found {len(ep_ids)} fixture episode(s): {ep_ids}")

    policy = SmolVLAPolicyAdapter(model_id=model_id, device="cpu")
    evaluator = OfflineEvaluator(
        policy,
        store,
        model_id=model_id,
        dataset_id="fixture",
        evaluation_mode=EvaluationMode.REPLAY,
    )
    result = evaluator.evaluate(ep_ids)
    _print_result(result)


def _print_result(result) -> None:  # type: ignore[type-arg]
    agg = result.aggregate
    print()
    print(f"  Evaluation mode : {result.evaluation_mode}")
    print(f"  Model           : {result.model_id}")
    print(f"  Dataset         : {result.dataset_id}")
    print(f"  Episodes        : {result.n_episodes}")
    print(f"  Total steps     : {agg.n_steps}")
    print(f"  L1 mean ± std   : {agg.l1_mean:.6f} ± {agg.l1_std:.6f}")
    print(f"  L2 mean ± std   : {agg.l2_mean:.6f} ± {agg.l2_std:.6f}")
    print()
    print(f"  NOTE: {result.evaluation_note}")
    print()
    for ep_res in result.per_episode:
        print(
            f"  [{ep_res.episode_id}] steps={ep_res.n_steps} "
            f"L1={ep_res.l1_mean:.4f} L2={ep_res.l2_mean:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline policy evaluation for LangGraph VLA Agent")
    parser.add_argument(
        "--mode",
        choices=["mock", "replay", "vla"],
        default="mock",
        help="Evaluation mode (default: mock)",
    )
    parser.add_argument(
        "--model-id",
        default="lerobot/smolvla_base",
        help="HuggingFace Hub model ID (vla mode only)",
    )
    args = parser.parse_args()

    if args.mode == "mock":
        _run_mock(args)
    elif args.mode == "replay":
        _run_replay(args)
    elif args.mode == "vla":
        _run_vla(args)


if __name__ == "__main__":
    main()
