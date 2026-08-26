"""Robot observation model."""

from typing import Any

import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, field_validator


class RobotObservation(BaseModel):
    """A single observation from the environment, passed to RobotPolicy.act().

    Fields
    ------
    state:
        1-D proprioceptive state vector (joint positions, velocities, etc.).
        Shape and units are embodiment-specific; for SO-100 this is 6 joint
        positions in radians.
    images:
        Camera observations keyed by camera name (e.g. "front", "wrist").
        Shape: H x W x C, dtype uint8. Empty dict is valid for mock/state-only
        environments.
    timestamp:
        Elapsed seconds since the start of the current episode.
    metadata:
        Optional structured extras (e.g. episode_id for replay scenarios).
        Not passed to the policy; useful for logging and evaluation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    state: npt.NDArray[Any]
    images: dict[str, npt.NDArray[Any]] = {}
    timestamp: float = 0.0
    metadata: dict[str, Any] = {}

    @field_validator("state")
    @classmethod
    def state_must_be_1d(cls, v: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if v.ndim != 1:
            raise ValueError(f"state must be a 1-D array, got shape {v.shape}")
        return v

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_non_negative(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError(f"timestamp must be >= 0, got {v}")
        return v
