"""Checkpoint comparison: offline evaluation of two policy checkpoints.

Compares base vs fine-tuned SmolVLA using the same OfflineEvaluator and the
same held-out episodes.  Works with any RobotPolicy implementation — testable
with stub policies in unit tests, and with SmolVLAPolicyAdapter in production.
"""

from __future__ import annotations

import structlog
from pydantic import BaseModel

from langgraph_vla_agent.datasets.store import EpisodeStore
from langgraph_vla_agent.evaluation.metrics import OfflineEvalResult
from langgraph_vla_agent.evaluation.offline import OfflineEvaluator
from langgraph_vla_agent.policies.base import RobotPolicy

_log = structlog.get_logger(__name__)

_COMPARISON_NOTE = (
    "Checkpoint comparison uses offline/replay evaluation. "
    "Delta metrics reflect action prediction error improvement on held-out episodes, "
    "not closed-loop task success."
)


class CheckpointComparisonResult(BaseModel):
    """Side-by-side offline evaluation of two checkpoints.

    delta_l1_mean = finetuned.aggregate.l1_mean - base.aggregate.l1_mean

    Interpretation
    --------------
    delta_l1_mean < 0  →  fine-tuned is closer to ground truth (improved)
    delta_l1_mean > 0  →  fine-tuned diverged further from ground truth (worse)
    improvement_pct_l1 > 0  →  improvement expressed as percentage of base error

    Critical limitation
    -------------------
    Lower action prediction error is a positive signal but does not guarantee
    improved closed-loop task success.  Every result carries evaluation_note.
    """

    base_result: OfflineEvalResult
    finetuned_result: OfflineEvalResult

    delta_l1_mean: float
    delta_l2_mean: float
    # (base_l1 - ft_l1) / base_l1 * 100; positive = improved
    improvement_pct_l1: float

    n_episodes: int
    dataset_id: str
    evaluation_note: str = _COMPARISON_NOTE

    def summary_lines(self) -> list[str]:
        """Human-readable comparison table lines."""
        b = self.base_result.aggregate
        f = self.finetuned_result.aggregate
        lines = [
            f"{'Metric':<22} {'Base':>10} {'Fine-tuned':>12} {'Delta':>10} {'Improv%':>9}",
            "-" * 66,
            (
                f"{'L1 mean':<22} {b.l1_mean:>10.6f} {f.l1_mean:>12.6f} "
                f"{self.delta_l1_mean:>+10.6f} {self.improvement_pct_l1:>+9.2f}%"
            ),
            (f"{'L2 mean':<22} {b.l2_mean:>10.6f} {f.l2_mean:>12.6f} {self.delta_l2_mean:>+10.6f}"),
            (f"{'L1 std':<22} {b.l1_std:>10.6f} {f.l1_std:>12.6f}"),
            (f"{'Steps evaluated':<22} {b.n_steps:>10} {f.n_steps:>12}"),
            "",
            f"Base model:       {self.base_result.model_id}",
            f"Fine-tuned model: {self.finetuned_result.model_id}",
            f"Episodes:         {self.n_episodes}  |  Dataset: {self.dataset_id}",
            "",
            f"Note: {self.evaluation_note}",
        ]
        return lines


def compare_checkpoints(
    base_policy: RobotPolicy,
    finetuned_policy: RobotPolicy,
    store: EpisodeStore,
    episode_ids: list[str],
    *,
    base_model_id: str,
    finetuned_model_id: str,
    dataset_id: str,
    run_id: str = "checkpoint-comparison",
) -> CheckpointComparisonResult:
    """Evaluate two policies on the same held-out episodes and return delta metrics.

    Both evaluations use identical episode_ids and the same EpisodeStore, so
    the comparison is apples-to-apples.  Uses OfflineEvaluator internally.

    Parameters
    ----------
    base_policy:
        The baseline policy (e.g. SmolVLAPolicyAdapter("lerobot/smolvla_base")).
    finetuned_policy:
        The fine-tuned policy (e.g. SmolVLAPolicyAdapter("artifacts/training/...")).
    store:
        Episode source for both evaluations.
    episode_ids:
        Held-out test-split episode IDs. Must not overlap with training episodes.
    base_model_id:
        Human-readable label for the base checkpoint (recorded in results).
    finetuned_model_id:
        Human-readable label for the fine-tuned checkpoint.
    dataset_id:
        Human-readable dataset label recorded in results.
    run_id:
        Identifier for this comparison run, used in log fields.
    """
    log = _log.bind(
        run_id=run_id,
        base=base_model_id,
        finetuned=finetuned_model_id,
        n_episodes=len(episode_ids),
    )
    log.info("comparison.start")

    base_result = OfflineEvaluator(
        base_policy,
        store,
        model_id=base_model_id,
        dataset_id=dataset_id,
    ).evaluate(episode_ids, run_id=f"{run_id}.base")

    finetuned_result = OfflineEvaluator(
        finetuned_policy,
        store,
        model_id=finetuned_model_id,
        dataset_id=dataset_id,
    ).evaluate(episode_ids, run_id=f"{run_id}.finetuned")

    base_l1 = base_result.aggregate.l1_mean
    ft_l1 = finetuned_result.aggregate.l1_mean
    delta_l1 = ft_l1 - base_l1
    delta_l2 = finetuned_result.aggregate.l2_mean - base_result.aggregate.l2_mean
    improvement_pct_l1 = ((base_l1 - ft_l1) / base_l1 * 100.0) if base_l1 > 0.0 else 0.0

    log.info(
        "comparison.done",
        base_l1=round(base_l1, 6),
        finetuned_l1=round(ft_l1, 6),
        delta_l1=round(delta_l1, 6),
        improvement_pct_l1=round(improvement_pct_l1, 2),
    )

    return CheckpointComparisonResult(
        base_result=base_result,
        finetuned_result=finetuned_result,
        delta_l1_mean=delta_l1,
        delta_l2_mean=delta_l2,
        improvement_pct_l1=improvement_pct_l1,
        n_episodes=len(episode_ids),
        dataset_id=dataset_id,
    )
