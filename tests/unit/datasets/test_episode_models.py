"""Unit tests for replay episode domain models (no network, no optional deps)."""

import pytest
from pydantic import ValidationError

from langgraph_vla_agent.datasets.episode import (
    DatasetProvenance,
    DatasetSplit,
    ReplayEpisode,
    ReplayStep,
)

# ---------------------------------------------------------------------------
# ReplayStep
# ---------------------------------------------------------------------------


def test_replay_step_valid() -> None:
    step = ReplayStep(
        timestep=0,
        observation={"state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]},
        action=[0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    assert step.timestep == 0
    assert step.terminated is False
    assert step.success is False


def test_replay_step_terminal_flags() -> None:
    step = ReplayStep(
        timestep=4,
        observation={"state": [0.0] * 6},
        action=[0.0] * 6,
        terminated=True,
        success=True,
    )
    assert step.terminated is True
    assert step.success is True
    assert step.truncated is False


# ---------------------------------------------------------------------------
# ReplayEpisode
# ---------------------------------------------------------------------------


def _make_episode(n_steps: int = 3, success: bool = True) -> ReplayEpisode:
    steps = [
        ReplayStep(
            timestep=i,
            observation={"state": [0.0] * 6},
            action=[0.0] * 6,
            terminated=(i == n_steps - 1),
            success=(success and i == n_steps - 1),
        )
        for i in range(n_steps)
    ]
    return ReplayEpisode(
        episode_id="ep-test",
        instruction="test instruction",
        dataset_id="fixture",
        action_dim=6,
        state_dim=6,
        steps=steps,
    )


def test_replay_episode_length_property() -> None:
    ep = _make_episode(n_steps=5)
    assert ep.length == 5


def test_replay_episode_empty_steps_raises() -> None:
    with pytest.raises(ValidationError, match="steps must not be empty"):
        ReplayEpisode(
            episode_id="ep-empty",
            instruction="test",
            dataset_id="fixture",
            action_dim=6,
            state_dim=6,
            steps=[],
        )


def test_replay_episode_instruction_preserved() -> None:
    ep = _make_episode()
    assert ep.instruction == "test instruction"


def test_replay_episode_last_step_is_terminal() -> None:
    ep = _make_episode(n_steps=3, success=True)
    assert ep.steps[-1].terminated is True
    assert ep.steps[-1].success is True


# ---------------------------------------------------------------------------
# DatasetSplit
# ---------------------------------------------------------------------------


def test_dataset_split_is_leak_free() -> None:
    split = DatasetSplit(
        train=["ep-1", "ep-2"],
        val=["ep-3"],
        test=["ep-4"],
        seed=42,
        train_ratio=0.5,
        val_ratio=0.25,
    )
    assert split.is_leak_free() is True


def test_dataset_split_detects_leakage() -> None:
    split = DatasetSplit(
        train=["ep-1", "ep-2"],
        val=["ep-2"],  # ep-2 appears in both train and val
        test=["ep-3"],
        seed=42,
        train_ratio=0.5,
        val_ratio=0.25,
    )
    assert split.is_leak_free() is False


def test_dataset_split_test_ratio() -> None:
    split = DatasetSplit(
        train=["ep-1"],
        val=["ep-2"],
        test=["ep-3"],
        seed=0,
        train_ratio=0.7,
        val_ratio=0.15,
    )
    assert abs(split.test_ratio - 0.15) < 1e-5


# ---------------------------------------------------------------------------
# DatasetProvenance
# ---------------------------------------------------------------------------


def test_dataset_provenance_model() -> None:
    prov = DatasetProvenance(
        hub_id="lerobot/svla_so100_pickplace",
        revision="abc123",
        license_spdx="Apache-2.0",
        episodes=50,
        embodiment="so100",
        action_dim=6,
        obs_keys=["observation.image.front", "observation.state"],
        language_field="task",
        notes="Reference dataset from SmolVLA paper.",
    )
    assert prov.hub_id == "lerobot/svla_so100_pickplace"
    assert prov.episodes == 50
    assert prov.action_dim == 6
