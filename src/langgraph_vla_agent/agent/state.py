"""AgentState — the durable LangGraph orchestration state."""

import operator
from enum import StrEnum
from typing import Annotated, TypedDict

from langgraph_vla_agent.domain.context import EvaluationMode
from langgraph_vla_agent.domain.results import ExecutionResult
from langgraph_vla_agent.domain.tasks import SubTask, TaskGoal, TaskPlan


class AgentStatus(StrEnum):
    """Terminal status of a full agent run (distinct from per-subtask ExecutionStatus)."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SAFETY_STOP = "safety_stop"


class AgentState(TypedDict):
    """LangGraph state for the VLA orchestration agent.

    Only durable orchestration metadata lives here — never raw observations,
    image tensors, or full trajectory arrays. Large data is referenced by
    opaque string IDs stored in execution_history_references.

    List fields annotated with operator.add use append semantics: a node
    returns only the NEW items and LangGraph appends them. All other fields
    use overwrite semantics: a node returns the complete new value.
    """

    # Identity
    run_id: str
    goal: TaskGoal

    # Planning
    plan: TaskPlan | None
    current_subtask: SubTask | None

    # Accumulator fields — nodes return only new items; LangGraph appends.
    completed_subtask_ids: Annotated[list[str], operator.add]
    failed_subtask_ids: Annotated[list[str], operator.add]
    execution_history_references: Annotated[list[str], operator.add]

    # Counters — overwrite semantics.
    retry_count: int
    replan_count: int

    # Execution
    last_execution_result: ExecutionResult | None

    # Safety
    safety_status: str  # "ok" | "rejected"
    safety_rejection_reason: str

    # Control
    evaluation_mode: EvaluationMode
    final_status: AgentStatus | None
    error_message: str


def make_initial_state(goal: TaskGoal) -> AgentState:
    """Return a fully-initialised AgentState for a new agent run."""
    return AgentState(
        run_id=goal.run_id,
        goal=goal,
        plan=None,
        current_subtask=None,
        completed_subtask_ids=[],
        failed_subtask_ids=[],
        execution_history_references=[],
        retry_count=0,
        replan_count=0,
        last_execution_result=None,
        safety_status="ok",
        safety_rejection_reason="",
        evaluation_mode=goal.evaluation_mode,
        final_status=None,
        error_message="",
    )
