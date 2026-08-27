"""Compare base vs fine-tuned SmolVLA on held-out episodes.

Runs OfflineEvaluator on two policy checkpoints against the same set of
episodes and prints a side-by-side comparison table.

Usage (see also: make compare-checkpoints):

    # Fixture mode — no GPU, no training required (uses stub policies):
    uv run python scripts/compare_checkpoints.py --mode fixture

    # VLA mode — requires [vla] extra and both checkpoints available:
    uv run python scripts/compare_checkpoints.py \\
        --base lerobot/smolvla_base \\
        --finetuned artifacts/training/smolvla_so100_m4 \\
        --mode vla

Fixture mode verifies that the comparison infrastructure works end-to-end
using zero-vector stub policies against the committed fixture episodes.
VLA mode is the production comparison run after actual fine-tuning.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langgraph_vla_agent.datasets.episode import ReplayEpisode
from langgraph_vla_agent.datasets.store import EpisodeStore
from langgraph_vla_agent.domain.context import PolicyContext
from langgraph_vla_agent.domain.observations import RobotObservation
from langgraph_vla_agent.evaluation.comparison import compare_checkpoints
from langgraph_vla_agent.policies.base import RobotPolicy


# ---------------------------------------------------------------------------
# Fixture mode helpers (no GPU required)
# ---------------------------------------------------------------------------

class _ZeroPolicy:
    """Stub that returns zero actions for all observations."""

    def __init__(self, name: str, action_dim: int = 6) -> None:
        import numpy as np
        self._name = name
        self._action_dim = action_dim
        self._np = np

    def reset(self, context: PolicyContext) -> None:
        pass

    def act(self, observation: RobotObservation, instruction: str):  # type: ignore[override]
        import numpy as np
        from langgraph_vla_agent.domain.actions import RobotAction
        return RobotAction(values=np.zeros(self._action_dim, dtype=np.float32))


class _FixtureStore:
    """Minimal EpisodeStore backed by JSON fixture files."""

    def __init__(self, fixture_dir: str) -> None:
        self._dir = Path(fixture_dir)

    def list_episode_ids(self) -> list[str]:
        return sorted(p.stem for p in self._dir.glob("*.json"))

    def load_episode(self, episode_id: str) -> ReplayEpisode:
        import json
        path = self._dir / f"{episode_id}.json"
        with open(path) as f:
            data = json.load(f)
        return ReplayEpisode.model_validate(data)


# ---------------------------------------------------------------------------
# VLA mode helpers (requires [vla] extra)
# ---------------------------------------------------------------------------

def _load_smolvla(model_id: str) -> RobotPolicy:
    try:
        from langgraph_vla_agent.policies.smolvla import SmolVLAPolicyAdapter
        return SmolVLAPolicyAdapter(model_id=model_id, device="cpu")
    except ImportError:
        print(
            "ERROR: lerobot not available. Install with:  make setup-vla",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two SmolVLA checkpoints offline")
    parser.add_argument(
        "--mode",
        choices=["fixture", "vla"],
        default="fixture",
        help="fixture: stub policies on fixture episodes (no GPU); vla: real models",
    )
    parser.add_argument(
        "--base",
        default="lerobot/smolvla_base",
        help="Base model ID (Hub ID or local path); used in vla mode",
    )
    parser.add_argument(
        "--finetuned",
        default=None,
        help="Fine-tuned model ID or path; used in vla mode",
    )
    parser.add_argument(
        "--fixture-dir",
        default="data/fixtures/episodes",
        help="Directory containing JSON fixture files (fixture mode only)",
    )
    args = parser.parse_args()

    if args.mode == "vla" and args.finetuned is None:
        parser.error("--finetuned is required in vla mode")

    if args.mode == "fixture":
        print("=== Checkpoint comparison — fixture mode (stub policies) ===")
        print("Both policies return zero actions. This verifies the comparison")
        print("infrastructure end-to-end without a GPU or trained model.\n")
        store: EpisodeStore = _FixtureStore(args.fixture_dir)
        episode_ids = store.list_episode_ids()
        base_policy: RobotPolicy = _ZeroPolicy("base-stub")
        finetuned_policy: RobotPolicy = _ZeroPolicy("finetuned-stub")
        base_id = "stub-base"
        ft_id = "stub-finetuned"
    else:
        print("=== Checkpoint comparison — VLA mode ===")
        print("Loading real SmolVLA checkpoints. This requires [vla] extra and model downloads.\n")
        store = _FixtureStore(args.fixture_dir)
        episode_ids = store.list_episode_ids()
        base_policy = _load_smolvla(args.base)
        finetuned_policy = _load_smolvla(args.finetuned)  # type: ignore[arg-type]
        base_id = args.base
        ft_id = args.finetuned  # type: ignore[assignment]

    print(f"Episodes: {episode_ids}\n")

    result = compare_checkpoints(
        base_policy=base_policy,
        finetuned_policy=finetuned_policy,
        store=store,
        episode_ids=episode_ids,
        base_model_id=base_id,
        finetuned_model_id=ft_id,
        dataset_id="fixture" if args.mode == "fixture" else "lerobot/svla_so100_pickplace",
    )

    print("\n=== Results ===")
    for line in result.summary_lines():
        print(line)


if __name__ == "__main__":
    main()
