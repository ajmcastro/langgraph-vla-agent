"""Fine-tune SmolVLA on a LeRobot dataset.

Validates the training config, generates the lerobot-train command, and
optionally executes it.  Records a provenance YAML on completion.

Usage (see also: make train):

    # Validate config + print lerobot-train command (no execution):
    uv run python scripts/train_smolvla.py --dry-run

    # Run training locally (requires GPU + vla extra):
    uv run python scripts/train_smolvla.py

    # Override the config file:
    uv run python scripts/train_smolvla.py --config configs/training/smolvla_so100.yaml

    # Override individual fields:
    uv run python scripts/train_smolvla.py --steps 500 --device mps

Requires: uv sync --extra dev --extra vla
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

# Allow running as a script without installing the package
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langgraph_vla_agent.training.config import SmolVLATrainingConfig, TrainingRunProvenance


def _load_config(path: str, overrides: dict[str, object]) -> SmolVLATrainingConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    raw.update(overrides)
    return SmolVLATrainingConfig.model_validate(raw)


def _check_vla_available() -> bool:
    try:
        import lerobot  # noqa: F401
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _save_provenance(
    cfg: SmolVLATrainingConfig,
    started_at: str,
    completed_at: str | None,
    final_step: int | None,
    error: str | None,
) -> Path:
    prov = TrainingRunProvenance(
        run_id=cfg.run_name,
        config=cfg,
        started_at=started_at,
        completed_at=completed_at,
        final_step=final_step,
        output_dir=f"{cfg.output_dir}/{cfg.run_name}",
        notes=f"error: {error}" if error else "",
    )
    out_dir = Path("data/provenance/training")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cfg.run_name}.yaml"
    prov_dict = json.loads(prov.model_dump_json())
    with open(out_path, "w") as f:
        yaml.safe_dump(prov_dict, f, default_flow_style=False, sort_keys=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune SmolVLA on a LeRobot dataset")
    parser.add_argument(
        "--config",
        default="configs/training/smolvla_so100.yaml",
        help="Path to training config YAML (default: configs/training/smolvla_so100.yaml)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print command without executing")
    parser.add_argument("--steps", type=int, help="Override steps")
    parser.add_argument("--batch-size", type=int, dest="batch_size", help="Override batch_size")
    parser.add_argument("--device", help="Override device (cuda|mps|cpu)")
    parser.add_argument("--seed", type=int, help="Override seed")
    args = parser.parse_args()

    overrides: dict[str, object] = {}
    if args.steps is not None:
        overrides["steps"] = args.steps
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.device is not None:
        overrides["device"] = args.device
    if args.seed is not None:
        overrides["seed"] = args.seed

    print(f"Loading config from: {args.config}")
    cfg = _load_config(args.config, overrides)

    print("\n=== SmolVLA Training Config (validated) ===")
    print(f"  run_name:         {cfg.run_name}")
    print(f"  base_model_id:    {cfg.base_model_id}")
    print(f"  dataset_id:       {cfg.dataset_id}")
    print(f"  dataset_revision: {cfg.dataset_revision or '(latest)'}")
    print(f"  batch_size:       {cfg.batch_size}")
    print(f"  steps:            {cfg.steps}")
    print(f"  seed:             {cfg.seed}")
    print(f"  device:           {cfg.device}")
    print(f"  eval_split:       {cfg.eval_split}")
    print(f"  output_dir:       {cfg.output_dir}/{cfg.run_name}")
    if cfg.hf_jobs_target:
        print(f"  hf_jobs_target:   {cfg.hf_jobs_target}  [REMOTE — will submit to HF Jobs]")
    else:
        print("  execution:        local")

    print("\n=== lerobot-train command ===")
    print(cfg.lerobot_train_command())

    if args.dry_run:
        print("\n[--dry-run] Exiting without executing.")
        return

    if not _check_vla_available():
        print(
            "\nERROR: lerobot / torch not available. "
            "Run:  make setup-vla  then retry.",
            file=sys.stderr,
        )
        sys.exit(1)

    started_at = datetime.now(UTC).isoformat()
    print(f"\n[{started_at}] Starting training…")

    cmd = ["lerobot-train"] + cfg.lerobot_train_args()
    error: str | None = None
    final_step: int | None = None
    exit_code = 0

    try:
        result = subprocess.run(cmd, check=True, env={**os.environ})
        final_step = cfg.steps
    except subprocess.CalledProcessError as e:
        error = str(e)
        exit_code = e.returncode
        print(f"\nTraining failed with exit code {exit_code}: {error}", file=sys.stderr)
    except KeyboardInterrupt:
        error = "interrupted"
        print("\nTraining interrupted.", file=sys.stderr)

    completed_at = datetime.now(UTC).isoformat() if error != "interrupted" else None
    prov_path = _save_provenance(cfg, started_at, completed_at, final_step, error)
    print(f"\nProvenance written to: {prov_path}")

    if exit_code != 0:
        sys.exit(exit_code)

    print("\nTraining complete. Next steps:")
    print("  1. Record GPU type, wall-clock time, and cost in the provenance YAML.")
    print("  2. Run:  make compare-checkpoints  to compare base vs fine-tuned.")


if __name__ == "__main__":
    main()
