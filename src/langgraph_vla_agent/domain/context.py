"""Episode context models: EvaluationMode and PolicyContext."""

from enum import StrEnum

from pydantic import BaseModel, Field


class EvaluationMode(StrEnum):
    """Identifies the execution backend for every result and metric.

    Every ExecutionResult, log line, and evaluation report must carry this value
    so that mock, replay, simulation, and hardware results are never conflated.
    """

    MOCK = "mock"
    REPLAY = "replay"
    SIMULATION = "simulation"
    HARDWARE = "hardware"


class PolicyContext(BaseModel):
    """Initialisation context passed to RobotPolicy.reset() at episode start.

    The policy uses this to set up any per-episode state (seed, logging handles,
    etc.) before the first call to act(). It carries no task-specific instruction
    — that arrives per-step via act(observation, instruction).
    """

    run_id: str
    episode_id: str
    evaluation_mode: EvaluationMode
    seed: int | None = None
    extra: dict[str, str] = Field(default_factory=dict)
