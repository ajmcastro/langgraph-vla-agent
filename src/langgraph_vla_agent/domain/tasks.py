"""Task models: SubTask, TaskGoal, TaskPlan."""

import uuid

from pydantic import BaseModel, Field

from langgraph_vla_agent.domain.context import EvaluationMode


class SubTask(BaseModel):
    """A single high-level manipulation step handed to the Executor.

    The LangGraph agent creates SubTasks from a TaskPlan. For M1 tests,
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
        Used by verification nodes and for logging.
    attempt:
        Retry counter incremented by the orchestration layer.
        Always 0 for the first attempt.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    instruction: str
    success_criteria: str = ""
    attempt: int = 0


class TaskGoal(BaseModel):
    """The natural-language goal provided by the operator.

    Fields
    ------
    text:
        Raw natural-language goal string (e.g. "pick up the cube and place it in the bin").
    run_id:
        Correlation ID for the full agent run; auto-generated if not provided.
    evaluation_mode:
        Propagated into every ExecutionResult for audit labelling.
    """

    text: str
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    evaluation_mode: EvaluationMode = EvaluationMode.MOCK


class TaskPlan(BaseModel):
    """An ordered list of SubTasks produced by a planner.

    Fields
    ------
    goal:
        The original goal this plan decomposes.
    subtasks:
        Ordered list of high-level manipulation steps.
    planner_id:
        Identifies which planner produced this plan (e.g. "deterministic/coarse").
    plan_version:
        Incremented each time the agent replans; 0 for the first plan.
    """

    goal: TaskGoal
    subtasks: list[SubTask]
    planner_id: str
    plan_version: int = 0

    def pending_subtasks(self, completed_ids: set[str], failed_ids: set[str]) -> list[SubTask]:
        """Return subtasks not yet completed or permanently failed."""
        done = completed_ids | failed_ids
        return [s for s in self.subtasks if s.id not in done]

    def is_complete(self, completed_ids: set[str]) -> bool:
        """True iff every subtask in this plan has been completed."""
        return all(s.id in completed_ids for s in self.subtasks)
