"""Execution result models: StepResult, ExecutionStatus, FailureReason, ExecutionResult."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from langgraph_vla_agent.domain.context import EvaluationMode


class ExecutionStatus(StrEnum):
    """Terminal status of an Executor.run() call."""

    SUCCESS = "success"
    FAILURE = "failure"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"
    TIMEOUT = "timeout"
    INVALID_ACTION = "invalid_action"
    SAFETY_STOP = "safety_stop"
    CANCELLED = "cancelled"
    ENVIRONMENT_ERROR = "environment_error"


class FailureReason(StrEnum):
    """Fine-grained cause when ExecutionStatus is not SUCCESS."""

    NONE = "none"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"
    TIMEOUT = "timeout"
    POLICY_ERROR = "policy_error"
    INVALID_ACTION = "invalid_action"
    ENVIRONMENT_ERROR = "environment_error"
    SAFETY_STOP = "safety_stop"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


class StepResult(BaseModel):
    """Returned by RobotEnvironment.step() after one action is applied.

    Fields
    ------
    terminated:
        The environment reached a natural terminal state (success or failure).
    truncated:
        The environment cut the episode short (time limit, workspace violation).
    success:
        True only when terminated=True and the subtask success predicate passed.
        Always False when terminated=False.
    info:
        Optional structured data for logging/debugging; not used by the Executor
        for control decisions.
    """

    terminated: bool
    truncated: bool
    success: bool
    info: dict[str, Any] = Field(default_factory=dict)


class ExecutionResult(BaseModel):
    """Returned by Executor.run() to the caller (LangGraph node in M5+).

    Intentionally does not contain raw observations or trajectory arrays.
    Large data is referenced by artifact_references, not embedded.
    """

    status: ExecutionStatus
    failure_reason: FailureReason
    steps_taken: int
    subtask_id: str
    evaluation_mode: EvaluationMode
    metrics: dict[str, float] = Field(default_factory=dict)
    artifact_references: list[str] = Field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.status == ExecutionStatus.SUCCESS
