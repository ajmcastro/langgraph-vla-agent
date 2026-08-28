"""LangGraph node functions and routing functions for the VLA agent graph.

Each node function takes AgentState and returns a partial-update dict.
LangGraph merges the dict into the current state (appending for Annotated
list fields, overwriting for plain fields).

Routing functions take AgentState and return the name of the next node
(or END from langgraph.graph). They are pure state → string mappings.

Dependency-injection pattern
-----------------------------
Nodes that need external objects (planner, executor, safety checker, config)
accept them as keyword-only arguments. graph.py wraps each such function in
a closure that closes over the concrete dependency — keeping these functions
unit-testable without building a full graph.
"""

from typing import Any

import structlog

from langgraph_vla_agent.agent.config import AgentConfig
from langgraph_vla_agent.agent.safety import SafetyChecker
from langgraph_vla_agent.agent.state import AgentState, AgentStatus
from langgraph_vla_agent.domain.context import PolicyContext
from langgraph_vla_agent.execution.executor import Executor
from langgraph_vla_agent.planning.base import PlanningError, TaskPlanner

_log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


def understand_goal(state: AgentState) -> dict[str, Any]:
    """Validate the goal text.  Sets final_status=FAILED on an empty goal."""
    log = _log.bind(run_id=state["run_id"])
    text = state["goal"].text.strip()
    if not text:
        log.error("agent.understand_goal.empty")
        return {"final_status": AgentStatus.FAILED, "error_message": "Goal text is empty"}
    log.info("agent.understand_goal", goal=text[:120])
    return {}


def create_plan(state: AgentState, *, planner: TaskPlanner) -> dict[str, Any]:
    """Call the planner to produce a TaskPlan.  Sets final_status=FAILED on PlanningError."""
    log = _log.bind(run_id=state["run_id"], replan_count=state["replan_count"])
    log.info("agent.create_plan")
    try:
        plan = planner.plan(state["goal"])
    except PlanningError as exc:
        log.error("agent.create_plan.failed", error=str(exc))
        return {"final_status": AgentStatus.FAILED, "error_message": str(exc)}
    log.info("agent.create_plan.ok", n_subtasks=len(plan.subtasks), planner=plan.planner_id)
    return {"plan": plan}


def select_next_subtask(state: AgentState) -> dict[str, Any]:
    """Pick the next pending subtask from the plan.

    Sets final_status=COMPLETED when all subtasks are done.
    Sets final_status=FAILED when no plan is available.
    """
    log = _log.bind(run_id=state["run_id"])
    plan = state["plan"]
    if plan is None:
        log.error("agent.select_next_subtask.no_plan")
        return {"final_status": AgentStatus.FAILED, "error_message": "No plan available"}

    completed = set(state["completed_subtask_ids"])
    failed = set(state["failed_subtask_ids"])
    pending = plan.pending_subtasks(completed, failed)

    if not pending:
        log.info("agent.select_next_subtask.all_done", total=len(plan.subtasks))
        return {"final_status": AgentStatus.COMPLETED, "current_subtask": None}

    subtask = pending[0]
    log.info(
        "agent.select_next_subtask",
        subtask_id=subtask.id[:8],
        instruction=subtask.instruction,
        remaining=len(pending),
    )
    return {"current_subtask": subtask}


def safety_check(state: AgentState, *, checker: SafetyChecker) -> dict[str, Any]:
    """Run the subtask instruction through the safety gate.

    Sets final_status=SAFETY_STOP on rejection.
    """
    subtask = state["current_subtask"]
    if subtask is None:
        return {
            "final_status": AgentStatus.FAILED,
            "error_message": "safety_check called with no current_subtask",
        }
    ok, reason = checker.check(subtask.instruction)
    if not ok:
        _log.bind(run_id=state["run_id"]).warning(
            "agent.safety_check.rejected",
            subtask_id=subtask.id[:8],
            reason=reason,
        )
        return {
            "safety_status": "rejected",
            "safety_rejection_reason": reason,
            "final_status": AgentStatus.SAFETY_STOP,
        }
    return {"safety_status": "ok", "safety_rejection_reason": ""}


def execute_policy(state: AgentState, *, executor: Executor) -> dict[str, Any]:
    """Run the Executor for the current subtask and record the result."""
    subtask = state["current_subtask"]
    assert subtask is not None, "execute_policy called with no current_subtask"

    context = PolicyContext(
        run_id=state["run_id"],
        episode_id=subtask.id,
        evaluation_mode=state["evaluation_mode"],
    )
    _log.bind(run_id=state["run_id"]).info(
        "agent.execute_policy",
        subtask_id=subtask.id[:8],
        attempt=subtask.attempt,
    )
    result = executor.run(subtask, context)
    ref = f"subtask:{subtask.id}:attempt:{subtask.attempt}"

    return {
        "last_execution_result": result,
        "execution_history_references": [ref],  # appended by LangGraph
    }


def verify_result(state: AgentState) -> dict[str, Any]:
    """Record subtask success; route to diagnose_failure on failure.

    On success: appends subtask id to completed_subtask_ids and resets retry_count.
    On failure: no state change (routing function decides next step).
    """
    result = state["last_execution_result"]
    subtask = state["current_subtask"]
    log = _log.bind(run_id=state["run_id"])

    if result is None or subtask is None:
        return {}

    if result.succeeded:
        log.info(
            "agent.verify_result.success",
            subtask_id=subtask.id[:8],
            steps=result.steps_taken,
        )
        return {
            "completed_subtask_ids": [subtask.id],  # appended by LangGraph
            "retry_count": 0,
        }

    log.info(
        "agent.verify_result.failure",
        subtask_id=subtask.id[:8],
        status=result.status,
        failure_reason=result.failure_reason,
    )
    return {}


def diagnose_failure(state: AgentState, *, config: AgentConfig) -> dict[str, Any]:
    """Decide whether to retry, replan, or permanently fail the episode.

    Decision logic:
    - retry_count < max_retries  →  increment retry_count, increment subtask.attempt
    - retry exhausted, replan_count < max_replans  →  replan (clear plan, record failed subtask)
    - both exhausted  →  final_status = FAILED
    """
    subtask = state["current_subtask"]
    log = _log.bind(run_id=state["run_id"])

    if state["retry_count"] < config.max_retries:
        new_count = state["retry_count"] + 1
        log.info("agent.diagnose.retry", retry_count=new_count, max=config.max_retries)
        updated = (
            subtask.model_copy(update={"attempt": subtask.attempt + 1}) if subtask else subtask
        )
        return {"retry_count": new_count, "current_subtask": updated}

    if state["replan_count"] < config.max_replans:
        new_replan = state["replan_count"] + 1
        log.info("agent.diagnose.replan", replan_count=new_replan, max=config.max_replans)
        failed_ids = [subtask.id] if subtask else []
        return {
            "replan_count": new_replan,
            "retry_count": 0,
            "failed_subtask_ids": failed_ids,  # appended by LangGraph
            "plan": None,
            "current_subtask": None,
        }

    log.warning("agent.diagnose.give_up", retries=config.max_retries, replans=config.max_replans)
    failed_ids = [subtask.id] if subtask else []
    return {
        "final_status": AgentStatus.FAILED,
        "failed_subtask_ids": failed_ids,
        "error_message": (
            f"Exhausted {config.max_retries} retries and {config.max_replans} replans"
        ),
    }


# ---------------------------------------------------------------------------
# Routing functions (pure state → node-name mappings)
# ---------------------------------------------------------------------------

_END = "__end__"  # matches langgraph.graph.END; avoids importing langgraph here


def route_after_understand(state: AgentState) -> str:
    return _END if state["final_status"] is not None else "create_plan"


def route_after_create_plan(state: AgentState) -> str:
    return _END if state["final_status"] is not None else "select_next_subtask"


def route_after_select(state: AgentState) -> str:
    return _END if state["final_status"] is not None else "safety_check"


def route_after_safety(state: AgentState) -> str:
    return _END if state["final_status"] is not None else "execute_policy"


def route_after_verify(state: AgentState) -> str:
    result = state["last_execution_result"]
    if result is not None and result.succeeded:
        return "select_next_subtask"
    return "diagnose_failure"


def route_after_diagnose(state: AgentState) -> str:
    if state["final_status"] is not None:
        return _END
    if state["plan"] is None:
        return "create_plan"
    return "execute_policy"
