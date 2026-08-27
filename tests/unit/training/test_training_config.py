"""Unit tests for SmolVLATrainingConfig and TrainingRunProvenance."""

import pytest
from pydantic import ValidationError

from langgraph_vla_agent.training.config import SmolVLATrainingConfig, TrainingRunProvenance

# ---------------------------------------------------------------------------
# SmolVLATrainingConfig — field validation
# ---------------------------------------------------------------------------


def test_default_config_is_valid() -> None:
    cfg = SmolVLATrainingConfig(run_name="test_run")
    assert cfg.base_model_id == "lerobot/smolvla_base"
    assert cfg.dataset_id == "lerobot/svla_so100_pickplace"
    assert cfg.batch_size == 8
    assert cfg.steps == 10_000
    assert cfg.seed == 42
    assert cfg.device == "cuda"


def test_run_name_is_required() -> None:
    with pytest.raises(ValidationError):
        SmolVLATrainingConfig.model_validate({})  # type: ignore[call-overload]


def test_rejects_zero_batch_size() -> None:
    with pytest.raises(ValidationError):
        SmolVLATrainingConfig(run_name="r", batch_size=0)


def test_rejects_negative_steps() -> None:
    with pytest.raises(ValidationError):
        SmolVLATrainingConfig(run_name="r", steps=-1)


def test_rejects_eval_split_ge_one() -> None:
    with pytest.raises(ValidationError):
        SmolVLATrainingConfig(run_name="r", eval_split=1.0)


def test_rejects_eval_split_negative() -> None:
    with pytest.raises(ValidationError):
        SmolVLATrainingConfig(run_name="r", eval_split=-0.1)


def test_eval_split_zero_is_valid() -> None:
    cfg = SmolVLATrainingConfig(run_name="r", eval_split=0.0)
    assert cfg.eval_split == 0.0


def test_train_episodes_none_by_default() -> None:
    cfg = SmolVLATrainingConfig(run_name="r")
    assert cfg.train_episodes is None


def test_train_episodes_list_accepted() -> None:
    cfg = SmolVLATrainingConfig(run_name="r", train_episodes=[0, 1, 2, 3, 4])
    assert cfg.train_episodes == [0, 1, 2, 3, 4]


def test_base_model_revision_defaults_to_none() -> None:
    cfg = SmolVLATrainingConfig(run_name="r")
    assert cfg.base_model_revision is None


# ---------------------------------------------------------------------------
# SmolVLATrainingConfig — lerobot_train_args / lerobot_train_command
# ---------------------------------------------------------------------------


def test_lerobot_train_args_contains_required_keys() -> None:
    cfg = SmolVLATrainingConfig(run_name="m4_test")
    args = cfg.lerobot_train_args()
    args_str = " ".join(args)
    assert "--dataset.repo_id=lerobot/svla_so100_pickplace" in args_str
    assert "--policy.type=smolvla" in args_str
    assert "--policy.pretrained_path=lerobot/smolvla_base" in args_str
    assert "--batch_size=8" in args_str
    assert "--steps=10000" in args_str
    assert "--seed=42" in args_str
    assert "--output_dir=artifacts/training/m4_test" in args_str


def test_lerobot_train_args_includes_push_to_hub_false_by_default() -> None:
    cfg = SmolVLATrainingConfig(run_name="r")
    args = " ".join(cfg.lerobot_train_args())
    assert "--policy.push_to_hub=false" in args


def test_lerobot_train_args_includes_push_to_hub_true_when_set() -> None:
    cfg = SmolVLATrainingConfig(run_name="r", push_to_hub=True, hf_entity="my-org")
    args = " ".join(cfg.lerobot_train_args())
    assert "--policy.push_to_hub=true" in args


def test_lerobot_train_args_all_have_double_dash_prefix() -> None:
    cfg = SmolVLATrainingConfig(run_name="r")
    args = cfg.lerobot_train_args()
    for arg in args:
        assert arg.startswith("--"), f"arg missing '--' prefix: {arg!r}"


def test_lerobot_train_args_includes_revision_when_set() -> None:
    cfg = SmolVLATrainingConfig(run_name="r", dataset_revision="abc123")
    args = " ".join(cfg.lerobot_train_args())
    assert "--dataset.revision=abc123" in args


def test_lerobot_train_args_omits_revision_when_none() -> None:
    cfg = SmolVLATrainingConfig(run_name="r")
    args = " ".join(cfg.lerobot_train_args())
    assert "dataset.revision" not in args


def test_lerobot_train_args_includes_episodes_when_set() -> None:
    cfg = SmolVLATrainingConfig(run_name="r", train_episodes=[0, 1, 2])
    args = " ".join(cfg.lerobot_train_args())
    assert "--dataset.episodes=[0,1,2]" in args


def test_lerobot_train_args_includes_rename_map_when_set() -> None:
    cfg = SmolVLATrainingConfig(
        run_name="r",
        rename_map={"observation.image.front": "observation.images.camera1"},
    )
    args = " ".join(cfg.lerobot_train_args())
    assert "--rename_map=" in args
    assert "observation.image.front" in args
    assert "observation.images.camera1" in args


def test_lerobot_train_args_omits_rename_map_when_empty() -> None:
    cfg = SmolVLATrainingConfig(run_name="r")
    args = " ".join(cfg.lerobot_train_args())
    assert "rename_map" not in args


def test_lerobot_train_args_includes_hf_jobs_when_set() -> None:
    cfg = SmolVLATrainingConfig(run_name="r", hf_jobs_target="nvidia-l40s-x1", hf_jobs_timeout="4h")
    args = " ".join(cfg.lerobot_train_args())
    assert "--job.target=nvidia-l40s-x1" in args
    assert "--job.timeout=4h" in args


def test_lerobot_train_command_is_multiline_string() -> None:
    cfg = SmolVLATrainingConfig(run_name="r")
    cmd = cfg.lerobot_train_command()
    assert cmd.startswith("lerobot-train")
    assert "\n" in cmd


# ---------------------------------------------------------------------------
# TrainingRunProvenance
# ---------------------------------------------------------------------------


def test_provenance_round_trip() -> None:
    cfg = SmolVLATrainingConfig(run_name="m4_test", steps=500)
    prov = TrainingRunProvenance(
        run_id="m4_test",
        config=cfg,
        started_at="2026-08-27T10:00:00+00:00",
    )
    as_json = prov.model_dump_json()
    restored = TrainingRunProvenance.model_validate_json(as_json)
    assert restored.run_id == "m4_test"
    assert restored.config.steps == 500
    assert restored.completed_at is None


def test_provenance_optional_fields_default_to_none() -> None:
    cfg = SmolVLATrainingConfig(run_name="r")
    prov = TrainingRunProvenance(run_id="r", config=cfg, started_at="2026-08-27T00:00:00+00:00")
    assert prov.gpu_type is None
    assert prov.test_l1_mean is None
    assert prov.delta_l1 is None
    assert prov.hub_model_id is None


def test_provenance_accepts_all_fields() -> None:
    cfg = SmolVLATrainingConfig(run_name="full")
    prov = TrainingRunProvenance(
        run_id="full",
        config=cfg,
        started_at="2026-08-27T08:00:00+00:00",
        completed_at="2026-08-27T10:30:00+00:00",
        final_step=10_000,
        best_val_step=8_000,
        best_val_loss=0.042,
        gpu_type="NVIDIA L40S",
        gpu_count=1,
        wall_clock_seconds=9000.0,
        approximate_cost_usd=7.50,
        output_dir="artifacts/training/full",
        hub_model_id="my-org/smolvla-so100-m4",
        test_l1_mean=0.120,
        test_l2_mean=0.155,
        base_l1_mean=0.176,
        delta_l1=-0.056,
        notes="First successful M4 run.",
    )
    assert prov.delta_l1 == pytest.approx(-0.056)
    assert prov.gpu_type == "NVIDIA L40S"
