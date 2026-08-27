"""Unit tests for CheckpointComparisonResult and compare_checkpoints()."""

import numpy as np
import pytest

from langgraph_vla_agent.datasets.episode import ReplayEpisode, ReplayStep
from langgraph_vla_agent.domain.actions import RobotAction
from langgraph_vla_agent.domain.context import PolicyContext
from langgraph_vla_agent.domain.observations import RobotObservation
from langgraph_vla_agent.evaluation.comparison import (
    CheckpointComparisonResult,
    compare_checkpoints,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ConstantPolicy:
    """Returns the same constant action every call."""

    def __init__(self, constant: np.ndarray) -> None:
        self._constant = constant.astype(np.float32)

    def reset(self, context: PolicyContext) -> None:
        pass

    def act(self, observation: RobotObservation, instruction: str) -> RobotAction:
        return RobotAction(values=self._constant.copy())


class _InMemoryStore:
    """EpisodeStore backed by a dict of ReplayEpisode objects."""

    def __init__(self, episodes: dict[str, ReplayEpisode]) -> None:
        self._episodes = episodes

    def list_episode_ids(self) -> list[str]:
        return list(self._episodes.keys())

    def load_episode(self, episode_id: str) -> ReplayEpisode:
        return self._episodes[episode_id]


def _make_episode(episode_id: str, action: list[float], n_steps: int = 3) -> ReplayEpisode:
    steps = []
    for i in range(n_steps):
        terminated = i == n_steps - 1
        steps.append(
            ReplayStep(
                timestep=i,
                observation={"state": [0.0] * len(action)},
                action=action,
                terminated=terminated,
                success=terminated,
            )
        )
    return ReplayEpisode(
        episode_id=episode_id,
        instruction="test instruction",
        dataset_id="test",
        action_dim=len(action),
        state_dim=len(action),
        steps=steps,
    )


# ---------------------------------------------------------------------------
# CheckpointComparisonResult — structure
# ---------------------------------------------------------------------------


def test_comparison_result_has_evaluation_note() -> None:
    gt = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0]
    store = _InMemoryStore({"ep1": _make_episode("ep1", gt)})

    zero = _ConstantPolicy(np.zeros(6))
    result = compare_checkpoints(
        base_policy=zero,
        finetuned_policy=zero,
        store=store,
        episode_ids=["ep1"],
        base_model_id="stub-base",
        finetuned_model_id="stub-ft",
        dataset_id="fixture",
    )
    assert result.evaluation_note != ""
    assert "offline" in result.evaluation_note.lower()


def test_comparison_result_is_pydantic_model() -> None:
    store = _InMemoryStore({"ep1": _make_episode("ep1", [0.0] * 6)})
    zero = _ConstantPolicy(np.zeros(6))
    result = compare_checkpoints(
        base_policy=zero,
        finetuned_policy=zero,
        store=store,
        episode_ids=["ep1"],
        base_model_id="a",
        finetuned_model_id="b",
        dataset_id="d",
    )
    as_json = result.model_dump_json()
    restored = CheckpointComparisonResult.model_validate_json(as_json)
    assert restored.delta_l1_mean == pytest.approx(result.delta_l1_mean)


# ---------------------------------------------------------------------------
# Delta metric correctness
# ---------------------------------------------------------------------------


def test_delta_is_zero_when_both_policies_identical() -> None:
    """Two identical policies on the same episodes → delta = 0."""
    gt = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    store = _InMemoryStore({"ep1": _make_episode("ep1", gt)})
    zero = _ConstantPolicy(np.zeros(6))

    result = compare_checkpoints(
        base_policy=zero,
        finetuned_policy=zero,
        store=store,
        episode_ids=["ep1"],
        base_model_id="same",
        finetuned_model_id="same",
        dataset_id="fixture",
    )
    assert result.delta_l1_mean == pytest.approx(0.0)
    assert result.delta_l2_mean == pytest.approx(0.0)
    assert result.improvement_pct_l1 == pytest.approx(0.0)


def test_negative_delta_when_finetuned_is_better() -> None:
    """Fine-tuned predicts the ground truth exactly; base predicts zeros.

    gt = [1, 0, 0, 0, 0, 0]
    base L1   = mean(|[0,0,...] - [1,0,...]|) = mean([1,0,0,0,0,0]) = 1/6
    ft   L1   = 0.0
    delta_l1  = 0.0 - 1/6 = -1/6  (negative = fine-tuned improved)
    """
    gt = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    store = _InMemoryStore({"ep1": _make_episode("ep1", gt, n_steps=1)})
    base_policy = _ConstantPolicy(np.zeros(6))
    ft_policy = _ConstantPolicy(np.array(gt))

    result = compare_checkpoints(
        base_policy=base_policy,
        finetuned_policy=ft_policy,
        store=store,
        episode_ids=["ep1"],
        base_model_id="base",
        finetuned_model_id="perfect-ft",
        dataset_id="fixture",
    )
    expected_base_l1 = 1.0 / 6.0
    assert result.base_result.aggregate.l1_mean == pytest.approx(expected_base_l1)
    assert result.finetuned_result.aggregate.l1_mean == pytest.approx(0.0)
    assert result.delta_l1_mean == pytest.approx(-expected_base_l1)
    assert result.improvement_pct_l1 == pytest.approx(100.0)


def test_positive_delta_when_finetuned_is_worse() -> None:
    """Fine-tuned moves further from gt than the base model.

    gt  = [1, 0, 0, 0, 0, 0]
    base predicts [0.5, 0, 0, 0, 0, 0]  → L1 = mean([0.5, 0, 0, 0, 0, 0]) = 0.5/6 ≈ 0.0833
    ft   predicts [2.0, 0, 0, 0, 0, 0]  → L1 = mean([1.0, 0, 0, 0, 0, 0]) = 1.0/6 ≈ 0.1667
    delta   = ft_l1 - base_l1 > 0  (fine-tuned is worse)
    improvement_pct_l1 < 0
    """
    gt = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    store = _InMemoryStore({"ep1": _make_episode("ep1", gt, n_steps=1)})
    base_policy = _ConstantPolicy(np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0]))
    ft_policy = _ConstantPolicy(np.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0]))

    result = compare_checkpoints(
        base_policy=base_policy,
        finetuned_policy=ft_policy,
        store=store,
        episode_ids=["ep1"],
        base_model_id="base",
        finetuned_model_id="worse-ft",
        dataset_id="fixture",
    )
    assert result.delta_l1_mean > 0.0
    assert result.improvement_pct_l1 < 0.0


def test_improvement_pct_zero_when_base_l1_is_zero() -> None:
    """Avoid division by zero when base error is already 0."""
    gt = [0.0] * 6
    store = _InMemoryStore({"ep1": _make_episode("ep1", gt, n_steps=1)})
    zero = _ConstantPolicy(np.zeros(6))

    result = compare_checkpoints(
        base_policy=zero,
        finetuned_policy=zero,
        store=store,
        episode_ids=["ep1"],
        base_model_id="perfect-base",
        finetuned_model_id="perfect-ft",
        dataset_id="fixture",
    )
    assert result.improvement_pct_l1 == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Multi-episode aggregation
# ---------------------------------------------------------------------------


def test_comparison_aggregates_multiple_episodes() -> None:
    """Both policies evaluated on 3 episodes; n_episodes recorded correctly."""
    episodes = {f"ep{i}": _make_episode(f"ep{i}", [float(i)] + [0.0] * 5) for i in range(3)}
    store = _InMemoryStore(episodes)
    zero = _ConstantPolicy(np.zeros(6))

    result = compare_checkpoints(
        base_policy=zero,
        finetuned_policy=zero,
        store=store,
        episode_ids=list(episodes.keys()),
        base_model_id="b",
        finetuned_model_id="f",
        dataset_id="fixture",
    )
    assert result.n_episodes == 3
    assert result.base_result.n_episodes == 3


# ---------------------------------------------------------------------------
# summary_lines
# ---------------------------------------------------------------------------


def test_summary_lines_contains_metric_headers() -> None:
    store = _InMemoryStore({"ep1": _make_episode("ep1", [0.0] * 6)})
    zero = _ConstantPolicy(np.zeros(6))
    result = compare_checkpoints(
        base_policy=zero,
        finetuned_policy=zero,
        store=store,
        episode_ids=["ep1"],
        base_model_id="a",
        finetuned_model_id="b",
        dataset_id="d",
    )
    lines = result.summary_lines()
    joined = "\n".join(lines)
    assert "L1 mean" in joined
    assert "L2 mean" in joined
    assert "Note:" in joined
