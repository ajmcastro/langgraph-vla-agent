"""Robot action model."""

from typing import Any

import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, field_validator


class RobotAction(BaseModel):
    """Action produced by a RobotPolicy and consumed by a RobotEnvironment.

    Fields
    ------
    values:
        1-D action vector. Semantics are embodiment-specific: for SmolVLA /
        SO-100 this is a chunk of absolute joint-position targets in radians.
        Shape must be (action_dim,); range validation is performed by the
        Executor before the action reaches the environment.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    values: npt.NDArray[Any]

    @field_validator("values")
    @classmethod
    def values_must_be_1d(cls, v: npt.NDArray[Any]) -> npt.NDArray[Any]:
        if v.ndim != 1:
            raise ValueError(f"values must be a 1-D array, got shape {v.shape}")
        return v
