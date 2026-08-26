"""Task models: SubTask (and stubs for M5 planning models)."""

import uuid

from pydantic import BaseModel, Field


class SubTask(BaseModel):
    """A single high-level manipulation step handed to the Executor.

    The LangGraph agent (M5+) creates SubTasks from a TaskPlan. For M1 tests,
    SubTasks are constructed directly.

    Fields
    ------
    id:
        Stable identifier; auto-generated if not provided.
    instruction:
        Natural-language description of the physical action to perform.
        Passed verbatim to RobotPolicy.act() at every step.
    success_criteria:
        Human-readable description of the terminal success condition.
        Used by verification nodes (M5+) and for logging.
    attempt:
        Retry counter incremented by the orchestration layer (M5+).
        Always 0 for the first attempt.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    instruction: str
    success_criteria: str = ""
    attempt: int = 0
