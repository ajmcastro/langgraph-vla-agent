"""Training configuration and provenance models for SmolVLA fine-tuning.

These Pydantic models are the project's source of truth for reproducibility.
scripts/train_smolvla.py validates a YAML config against SmolVLATrainingConfig
and translates it into the lerobot-train CLI invocation.

No GPU or ML deps are required to import this module.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field


class SmolVLATrainingConfig(BaseModel):
    """All hyperparameters needed to reproduce a SmolVLA fine-tuning run.

    Validated from configs/training/*.yaml by scripts/train_smolvla.py.
    Every field corresponds to a lerobot-train CLI argument or a project-level
    setting recorded for provenance.

    Data leakage note
    -----------------
    eval_split controls the fraction of dataset episodes held out for online
    val-loss monitoring *during* training (handled inside lerobot). The test
    split used for the final base-vs-fine-tuned comparison is separate — it
    must not overlap with training episodes. For svla_so100_pickplace (50
    episodes), the convention is: train=0.70, val=0.15 (eval_split), test=0.15.
    """

    # Human-readable identifier for this run; used as output sub-directory.
    run_name: str

    # Model
    base_model_id: str = "lerobot/smolvla_base"
    # Pin to a specific Hub commit for bit-exact reproducibility.
    # None = latest at training time (record the resolved hash in provenance).
    base_model_revision: str | None = None

    # Dataset
    dataset_id: str = "lerobot/svla_so100_pickplace"
    dataset_revision: str | None = None
    # Integer episode indices to use for training. None = all minus eval_split.
    # Example: [0, 1, ..., 34] for the first 35 of 50 episodes.
    train_episodes: list[int] | None = None
    # Fraction of dataset episodes held out for val-loss monitoring inside lerobot.
    # 0.15 ≈ 7-8 episodes of the 50-episode svla_so100_pickplace dataset.
    eval_split: float = Field(default=0.15, ge=0.0, lt=1.0)

    # Training hyperparameters
    batch_size: int = Field(default=8, gt=0)
    steps: int = Field(default=10_000, gt=0)
    seed: int = 42
    # Save a checkpoint every N steps (and at the final step).
    save_freq: int = Field(default=2_500, gt=0)

    # Hardware
    device: str = "cuda"  # "cuda" | "mps" | "cpu"

    # Output
    output_dir: str = "artifacts/training"
    # Push each saved checkpoint to HF Hub during training.
    push_to_hub: bool = False
    # HF org or user for push_to_hub. Required when push_to_hub=True.
    hf_entity: str | None = None

    # Logging
    wandb_enable: bool = False
    wandb_project: str = "langgraph-vla-agent"

    # Feature key remapping: maps dataset observation keys to the names the
    # policy expects.  Required when the dataset and checkpoint use different
    # camera key conventions.
    # Example for svla_so100_pickplace → smolvla_base:
    #   rename_map: {"observation.image.front": "observation.images.camera1"}
    rename_map: dict[str, str] = Field(default_factory=dict)

    # HF Jobs — remote cloud GPU execution.
    # None = run locally. Set to a flavor string (e.g. "nvidia-l40s-x1") to
    # submit to HF Jobs instead of running on this machine.
    hf_jobs_target: str | None = None
    hf_jobs_timeout: str = "8h"

    def lerobot_train_args(self) -> list[str]:
        """Generate the argument list for the lerobot-train CLI.

        Uses draccus ``--key=value`` format (double-dash prefix required).
        Run ``lerobot-train --help`` for the full argument reference.

        Key args:
          --policy.type=smolvla          — select SmolVLA architecture
          --policy.pretrained_path=...   — fine-tune from this Hub ID or path
          --dataset.repo_id=...
          --rename_map=...               — remap dataset keys to policy keys
        """
        run_output = f"{self.output_dir}/{self.run_name}"
        push = str(self.push_to_hub).lower()
        args: list[str] = [
            "--policy.type=smolvla",
            f"--policy.pretrained_path={self.base_model_id}",
            f"--dataset.repo_id={self.dataset_id}",
            f"--dataset.eval_split={self.eval_split}",
            f"--batch_size={self.batch_size}",
            f"--steps={self.steps}",
            f"--seed={self.seed}",
            f"--save_freq={self.save_freq}",
            f"--output_dir={run_output}",
            f"--policy.push_to_hub={push}",
        ]
        if self.dataset_revision:
            args.append(f"--dataset.revision={self.dataset_revision}")
        if self.base_model_revision:
            args.append(f"--policy.pretrained_revision={self.base_model_revision}")
        if self.train_episodes is not None:
            ep_str = "[" + ",".join(str(e) for e in self.train_episodes) + "]"
            args.append(f"--dataset.episodes={ep_str}")
        if self.rename_map:
            args.append(f"--rename_map={json.dumps(self.rename_map)}")
        if self.wandb_enable:
            args.extend(
                [
                    "--wandb.enable=true",
                    f"--wandb.project={self.wandb_project}",
                ]
            )
        if self.hf_jobs_target:
            args.extend(
                [
                    f"--job.target={self.hf_jobs_target}",
                    f"--job.timeout={self.hf_jobs_timeout}",
                ]
            )
        return args

    def lerobot_train_command(self) -> str:
        """Return the full lerobot-train shell command as a human-readable string."""
        args = self.lerobot_train_args()
        lines = ["lerobot-train \\"] + [f"  {a} \\" for a in args[:-1]] + [f"  {args[-1]}"]
        return "\n".join(lines)


class TrainingRunProvenance(BaseModel):
    """Record of a completed (or in-progress) training run.

    Committed to data/provenance/training/<run_name>.yaml after a run completes.
    Large artifacts (checkpoints, model weights) are never committed — only
    path references and aggregate metrics.
    """

    run_id: str  # matches config.run_name
    config: SmolVLATrainingConfig

    # Timeline (ISO-8601 strings)
    started_at: str
    completed_at: str | None = None

    # Training outcome
    final_step: int | None = None
    best_val_step: int | None = None
    best_val_loss: float | None = None

    # Hardware (filled in after the run)
    gpu_type: str | None = None  # e.g. "NVIDIA L40S"
    gpu_count: int | None = None
    wall_clock_seconds: float | None = None
    approximate_cost_usd: float | None = None

    # Artifact references (paths or Hub IDs, not data)
    output_dir: str | None = None
    hub_model_id: str | None = None  # if pushed to HF Hub

    # Offline evaluation on held-out test set
    # Run compare_checkpoints() after training to fill these in.
    test_l1_mean: float | None = None
    test_l2_mean: float | None = None
    base_l1_mean: float | None = None
    delta_l1: float | None = None  # test_l1 - base_l1; negative = improved

    notes: str = ""
