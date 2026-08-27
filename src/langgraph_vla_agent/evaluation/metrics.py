"""Evaluation result models for offline action-prediction assessment."""

from pydantic import BaseModel, Field

from langgraph_vla_agent.domain.context import EvaluationMode

_OFFLINE_NOTE = (
    "Offline/replay evaluation — results cannot be extrapolated to closed-loop task performance."
)


class ActionErrorMetrics(BaseModel):
    """Aggregate action prediction error across all evaluated steps.

    Both L1 and L2 are per-step scalars (mean over action dimensions),
    then aggregated (mean ± std) across all steps in all evaluated episodes.

    Interpretation
    --------------
    l1_mean / l2_mean:
        Lower is better. Zero means the policy predicted the exact recorded
        action at every step (achievable with ReplayRobotPolicy as the baseline).
    Limitation:
        Low error does NOT imply task success in closed-loop execution.
        The replay environment serves recorded observations regardless of
        what action the policy takes.
    """

    l1_mean: float = Field(ge=0.0)
    l1_std: float = Field(ge=0.0)
    l2_mean: float = Field(ge=0.0)
    l2_std: float = Field(ge=0.0)
    n_steps: int = Field(ge=0)


class EpisodeEvalResult(BaseModel):
    """Per-episode evaluation result: per-step action errors."""

    episode_id: str
    instruction: str
    n_steps: int
    action_errors_l1: list[float]
    action_errors_l2: list[float]

    @property
    def l1_mean(self) -> float:
        if not self.action_errors_l1:
            return 0.0
        return sum(self.action_errors_l1) / len(self.action_errors_l1)

    @property
    def l2_mean(self) -> float:
        if not self.action_errors_l2:
            return 0.0
        return sum(self.action_errors_l2) / len(self.action_errors_l2)


class OfflineEvalResult(BaseModel):
    """Full offline evaluation result for one policy on one set of episodes.

    Every result carries an evaluation_note that states the fundamental
    limitation of offline replay evaluation. This is structural — it is
    impossible to produce an OfflineEvalResult without this note.
    """

    evaluation_mode: EvaluationMode
    model_id: str
    dataset_id: str
    n_episodes: int
    aggregate: ActionErrorMetrics
    per_episode: list[EpisodeEvalResult]
    evaluation_note: str = _OFFLINE_NOTE
