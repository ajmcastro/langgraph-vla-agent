"""Unit tests for evaluation.metrics models."""

import pytest

from langgraph_vla_agent.domain.context import EvaluationMode
from langgraph_vla_agent.evaluation.metrics import (
    ActionErrorMetrics,
    EpisodeEvalResult,
    OfflineEvalResult,
)

# ---------------------------------------------------------------------------
# ActionErrorMetrics
# ---------------------------------------------------------------------------


def test_action_error_metrics_valid_zeros() -> None:
    m = ActionErrorMetrics(l1_mean=0.0, l1_std=0.0, l2_mean=0.0, l2_std=0.0, n_steps=0)
    assert m.l1_mean == 0.0
    assert m.n_steps == 0


def test_action_error_metrics_valid_nonzero() -> None:
    m = ActionErrorMetrics(l1_mean=0.05, l1_std=0.01, l2_mean=0.07, l2_std=0.02, n_steps=12)
    assert m.l1_mean == pytest.approx(0.05)
    assert m.n_steps == 12


def test_action_error_metrics_rejects_negative_mean() -> None:
    with pytest.raises(ValueError):
        ActionErrorMetrics(l1_mean=-0.1, l1_std=0.0, l2_mean=0.0, l2_std=0.0, n_steps=5)


def test_action_error_metrics_rejects_negative_std() -> None:
    with pytest.raises(ValueError):
        ActionErrorMetrics(l1_mean=0.0, l1_std=-0.5, l2_mean=0.0, l2_std=0.0, n_steps=5)


def test_action_error_metrics_rejects_negative_n_steps() -> None:
    with pytest.raises(ValueError):
        ActionErrorMetrics(l1_mean=0.0, l1_std=0.0, l2_mean=0.0, l2_std=0.0, n_steps=-1)


# ---------------------------------------------------------------------------
# EpisodeEvalResult
# ---------------------------------------------------------------------------


def _ep_result(l1s: list[float], l2s: list[float]) -> EpisodeEvalResult:
    return EpisodeEvalResult(
        episode_id="ep-001",
        instruction="pick up the block",
        n_steps=len(l1s),
        action_errors_l1=l1s,
        action_errors_l2=l2s,
    )


def test_episode_eval_result_l1_mean() -> None:
    r = _ep_result([0.1, 0.3], [0.2, 0.4])
    assert r.l1_mean == pytest.approx(0.2)


def test_episode_eval_result_l2_mean() -> None:
    r = _ep_result([0.1, 0.3], [0.2, 0.4])
    assert r.l2_mean == pytest.approx(0.3)


def test_episode_eval_result_empty_errors_return_zero() -> None:
    r = _ep_result([], [])
    assert r.l1_mean == 0.0
    assert r.l2_mean == 0.0


def test_episode_eval_result_n_steps_matches_errors() -> None:
    r = _ep_result([0.1, 0.2, 0.3], [0.1, 0.2, 0.3])
    assert r.n_steps == 3


# ---------------------------------------------------------------------------
# OfflineEvalResult
# ---------------------------------------------------------------------------


def _make_offline_result() -> OfflineEvalResult:
    ep = _ep_result([0.1, 0.2], [0.15, 0.25])
    agg = ActionErrorMetrics(l1_mean=0.15, l1_std=0.05, l2_mean=0.2, l2_std=0.05, n_steps=2)
    return OfflineEvalResult(
        evaluation_mode=EvaluationMode.REPLAY,
        model_id="test-model",
        dataset_id="fixture",
        n_episodes=1,
        aggregate=agg,
        per_episode=[ep],
    )


def test_offline_result_carries_evaluation_note() -> None:
    result = _make_offline_result()
    assert "closed-loop" in result.evaluation_note


def test_offline_result_evaluation_mode_is_replay() -> None:
    result = _make_offline_result()
    assert result.evaluation_mode == EvaluationMode.REPLAY


def test_offline_result_n_episodes_matches_per_episode() -> None:
    result = _make_offline_result()
    assert result.n_episodes == len(result.per_episode)


def test_offline_result_aggregate_access() -> None:
    result = _make_offline_result()
    assert result.aggregate.l1_mean == pytest.approx(0.15)
