"""Replay episode domain models — data containers for recorded trajectories."""

from pydantic import BaseModel, Field, field_validator


class ReplayStep(BaseModel):
    """One recorded step from a robot trajectory.

    Fields
    ------
    timestep:
        Zero-based index within the episode.
    observation:
        Observation dict keyed by modality name (e.g. ``"state"``).
        Values are lists of floats (JSON-serialisable). Camera images are
        omitted from fixture files; the replay environment converts
        ``"state"`` to a numpy array for RobotObservation.
    action:
        Joint-position delta or absolute command applied at this timestep.
        Length must equal the episode's ``action_dim``.
    terminated:
        True if the episode ended naturally (success or failure) at this step.
    truncated:
        True if the episode was cut short (step budget, workspace limit, etc.).
    success:
        True if the task success predicate was satisfied at this step.
        Meaningful only when ``terminated`` is True.
    """

    timestep: int
    observation: dict[str, list[float]]
    action: list[float]
    terminated: bool = False
    truncated: bool = False
    success: bool = False


class ReplayEpisode(BaseModel):
    """A complete recorded episode: a sequence of ReplayStep objects.

    Used by ReplayRobotPolicy (to serve recorded actions) and by
    ReplayEnvironment (to serve recorded observations and terminal signals).
    Both receive the same ReplayEpisode reference and maintain independent
    step pointers.
    """

    episode_id: str
    instruction: str
    dataset_id: str
    action_dim: int = Field(gt=0)
    state_dim: int = Field(gt=0)
    steps: list[ReplayStep]

    @field_validator("steps")
    @classmethod
    def steps_must_not_be_empty(cls, v: list[ReplayStep]) -> list[ReplayStep]:
        if not v:
            raise ValueError(
                "steps must not be empty — a replay episode must have at least one step"
            )
        return v

    @property
    def length(self) -> int:
        """Number of recorded steps in this episode."""
        return len(self.steps)


class DatasetProvenance(BaseModel):
    """Provenance record for a dataset used in experiments.

    Committed alongside evaluation results to ensure reproducibility.
    Corresponds to the YAML schema in data/provenance/.
    """

    hub_id: str
    revision: str | None = None
    license_spdx: str | None = None
    episodes: int = Field(gt=0)
    embodiment: str
    action_dim: int = Field(gt=0)
    obs_keys: list[str]
    language_field: str
    download_date: str | None = None
    notes: str = ""


class DatasetSplit(BaseModel):
    """A deterministic train/val/test partition of episode IDs.

    Produced by EpisodeSplitter. The seed and ratios are recorded for
    reproducibility — given the same episode list, seed, and ratios,
    EpisodeSplitter will produce the same DatasetSplit.
    """

    train: list[str]
    val: list[str]
    test: list[str]
    seed: int
    train_ratio: float
    val_ratio: float

    @property
    def test_ratio(self) -> float:
        return round(1.0 - self.train_ratio - self.val_ratio, 6)

    def is_leak_free(self) -> bool:
        """Return True if no episode_id appears in more than one partition."""
        train_s, val_s, test_s = set(self.train), set(self.val), set(self.test)
        return train_s.isdisjoint(val_s) and train_s.isdisjoint(test_s) and val_s.isdisjoint(test_s)
