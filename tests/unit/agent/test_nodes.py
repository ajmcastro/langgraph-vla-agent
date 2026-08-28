"""Unit tests for individual node functions and routing functions.

Node functions are tested by constructing plain AgentState dicts and calling
the function directly — no LangGraph import needed.  Dependencies (planner,
executor, config, checker) are injected as keyword arguments or stubs.
"""

from typing import Any
from unittest.mock import MagicMock

from langgraph_vla_agent.agent.config import AgentConfig
from langgraph_vla_agent.agent.nodes import (
    create_plan,
    diagnose_failure,
    route_after_create_plan,
    route_after_diagnose,
    route_after_select,
    route_after_verify,
    safety_check,
    select_next_subtask,
    understand_goal,
    verify_result,
)
from langgraph_vla_agent.agent.safety import SafetyChecker
from langgraph_vla_agent.agent.state import AgentState, AgentStatus, make_initial_state
from langgraph_vla_agent.domain.context import EvaluationMode
from langgraph_vla_agent.domain.results import ExecutionResult, ExecutionStatus, FailureReason
from langgraph_vla_agent.domain.tasks import SubTask, TaskGoal, TaskPlan
from langgraph_vla_agent.planning.base import PlanningError
from langgraph_vla_agent.planning.deterministic import DeterministicPlanner

_END = "__end__"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _goal(text: str = "pick up the cube and place it in the bin") -> TaskGoal:
    return TaskGoal(text=text, run_id="test-run", evaluation_mode=EvaluationMode.MOCK)


def _base_state(**overrides: Any) -> AgentState:
    state = make_initial_state(_goal())
    state.update(overrides)  # type: ignore[attr-defined]
    return state


def _subtask(instruction: str = "pick up the object") -> SubTask:
    return SubTask(instruction=instruction, success_criteria="done")


def _exec_result(succeeded: bool) -> ExecutionResult:
    return ExecutionResult(
        status=ExecutionStatus.SUCCESS if succeeded else ExecutionStatus.FAILURE,
        failure_reason=FailureReason.NONE if succeeded else FailureReason.UNKNOWN,
        steps_taken=3,
        subtask_id="st-001",
        evaluation_mode=EvaluationMode.MOCK,
    )


# ---------------------------------------------------------------------------
# understand_goal
# ---------------------------------------------------------------------------


def test_understand_goal_passes_valid_goal() -> None:
    state = _base_state()
    result = understand_goal(state)
    assert result == {} or result.get("final_status") is None


def test_understand_goal_fails_empty_goal() -> None:
    state = _base_state(goal=_goal(""))
    result = understand_goal(state)
    assert result["final_status"] == AgentStatus.FAILED


def test_understand_goal_fails_whitespace_only() -> None:
    state = _base_state(goal=_goal("   "))
    result = understand_goal(state)
    assert result["final_status"] == AgentStatus.FAILED


# ---------------------------------------------------------------------------
# create_plan
# ---------------------------------------------------------------------------


def test_create_plan_produces_plan() -> None:
    state = _base_state()
    planner = DeterministicPlanner(granularity="coarse")
    result = create_plan(state, planner=planner)
    assert "plan" in result
    assert len(result["plan"].subtasks) == 2


def test_create_plan_fails_on_unknown_goal() -> None:
    state = _base_state(goal=_goal("dance around the room"))
    planner = DeterministicPlanner()
    result = create_plan(state, planner=planner)
    assert result["final_status"] == AgentStatus.FAILED
    assert "error_message" in result


def test_create_plan_uses_injected_planner() -> None:
    state = _base_state()
    stub = MagicMock()
    stub.plan.return_value = TaskPlan(
        goal=state["goal"],
        subtasks=[_subtask()],
        planner_id="stub",
    )
    result = create_plan(state, planner=stub)
    assert result["plan"].planner_id == "stub"
    stub.plan.assert_called_once()


def test_create_plan_wraps_planning_error() -> None:
    state = _base_state()
    stub = MagicMock()
    stub.plan.side_effect = PlanningError("oops")
    result = create_plan(state, planner=stub)
    assert result["final_status"] == AgentStatus.FAILED


# ---------------------------------------------------------------------------
# select_next_subtask
# ---------------------------------------------------------------------------


def test_select_next_subtask_returns_first_pending() -> None:
    st = _subtask()
    plan = TaskPlan(goal=_goal(), subtasks=[st], planner_id="test")
    state = _base_state(plan=plan)
    result = select_next_subtask(state)
    assert result["current_subtask"].id == st.id


def test_select_next_subtask_skips_completed() -> None:
    st1 = _subtask("pick up")
    st2 = _subtask("place down")
    plan = TaskPlan(goal=_goal(), subtasks=[st1, st2], planner_id="test")
    state = _base_state(plan=plan, completed_subtask_ids=[st1.id])
    result = select_next_subtask(state)
    assert result["current_subtask"].id == st2.id


def test_select_next_subtask_sets_completed_when_all_done() -> None:
    st = _subtask()
    plan = TaskPlan(goal=_goal(), subtasks=[st], planner_id="test")
    state = _base_state(plan=plan, completed_subtask_ids=[st.id])
    result = select_next_subtask(state)
    assert result["final_status"] == AgentStatus.COMPLETED


def test_select_next_subtask_fails_with_no_plan() -> None:
    state = _base_state(plan=None)
    result = select_next_subtask(state)
    assert result["final_status"] == AgentStatus.FAILED


# ---------------------------------------------------------------------------
# safety_check
# ---------------------------------------------------------------------------


def test_safety_check_passes_valid_instruction() -> None:
    state = _base_state(current_subtask=_subtask("pick up the object"))
    result = safety_check(state, checker=SafetyChecker())
    assert result["safety_status"] == "ok"
    assert result["safety_rejection_reason"] == ""


def test_safety_check_rejects_blocked_instruction() -> None:
    state = _base_state(current_subtask=_subtask("move toward the human"))
    result = safety_check(state, checker=SafetyChecker())
    assert result["safety_status"] == "rejected"
    assert result["final_status"] == AgentStatus.SAFETY_STOP


def test_safety_check_fails_when_no_subtask() -> None:
    state = _base_state(current_subtask=None)
    result = safety_check(state, checker=SafetyChecker())
    assert result["final_status"] == AgentStatus.FAILED


# ---------------------------------------------------------------------------
# verify_result
# ---------------------------------------------------------------------------


def test_verify_result_records_success() -> None:
    st = _subtask()
    state = _base_state(current_subtask=st, last_execution_result=_exec_result(True))
    result = verify_result(state)
    assert result["completed_subtask_ids"] == [st.id]
    assert result["retry_count"] == 0


def test_verify_result_returns_empty_dict_on_failure() -> None:
    st = _subtask()
    state = _base_state(current_subtask=st, last_execution_result=_exec_result(False))
    result = verify_result(state)
    assert "completed_subtask_ids" not in result
    assert "final_status" not in result


def test_verify_result_handles_none_result() -> None:
    state = _base_state()
    result = verify_result(state)
    assert result == {}


# ---------------------------------------------------------------------------
# diagnose_failure
# ---------------------------------------------------------------------------


def test_diagnose_increments_retry_when_retries_remain() -> None:
    config = AgentConfig(max_retries=2, max_replans=1)
    st = _subtask()
    state = _base_state(current_subtask=st, retry_count=0)
    result = diagnose_failure(state, config=config)
    assert result["retry_count"] == 1
    assert result["current_subtask"].attempt == 1


def test_diagnose_increments_attempt_on_retry() -> None:
    config = AgentConfig(max_retries=3, max_replans=0)
    st = SubTask(instruction="pick", attempt=1)
    state = _base_state(current_subtask=st, retry_count=1)
    result = diagnose_failure(state, config=config)
    assert result["current_subtask"].attempt == 2


def test_diagnose_triggers_replan_after_max_retries() -> None:
    config = AgentConfig(max_retries=2, max_replans=1)
    st = _subtask()
    state = _base_state(current_subtask=st, retry_count=2)
    result = diagnose_failure(state, config=config)
    assert result["plan"] is None
    assert result["replan_count"] == 1
    assert st.id in result["failed_subtask_ids"]


def test_diagnose_fails_after_max_retries_and_replans() -> None:
    config = AgentConfig(max_retries=1, max_replans=0)
    st = _subtask()
    state = _base_state(current_subtask=st, retry_count=1, replan_count=0)
    result = diagnose_failure(state, config=config)
    assert result["final_status"] == AgentStatus.FAILED
    assert "error_message" in result


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------


def test_route_after_create_plan_proceeds_when_no_final_status() -> None:
    state = _base_state()
    assert route_after_create_plan(state) == "select_next_subtask"


def test_route_after_create_plan_ends_on_final_status() -> None:
    state = _base_state(final_status=AgentStatus.FAILED)
    assert route_after_create_plan(state) == _END


def test_route_after_select_proceeds_when_subtask_set() -> None:
    st = _subtask()
    state = _base_state(current_subtask=st)
    assert route_after_select(state) == "safety_check"


def test_route_after_select_ends_when_completed() -> None:
    state = _base_state(final_status=AgentStatus.COMPLETED)
    assert route_after_select(state) == _END


def test_route_after_verify_goes_to_select_on_success() -> None:
    st = _subtask()
    state = _base_state(current_subtask=st, last_execution_result=_exec_result(True))
    assert route_after_verify(state) == "select_next_subtask"


def test_route_after_verify_goes_to_diagnose_on_failure() -> None:
    st = _subtask()
    state = _base_state(current_subtask=st, last_execution_result=_exec_result(False))
    assert route_after_verify(state) == "diagnose_failure"


def test_route_after_diagnose_retries_when_plan_exists() -> None:
    st = _subtask()
    plan = TaskPlan(goal=_goal(), subtasks=[st], planner_id="t")
    state = _base_state(plan=plan)
    assert route_after_diagnose(state) == "execute_policy"


def test_route_after_diagnose_replans_when_plan_is_none() -> None:
    state = _base_state(plan=None)
    assert route_after_diagnose(state) == "create_plan"


def test_route_after_diagnose_ends_on_final_status() -> None:
    state = _base_state(final_status=AgentStatus.FAILED)
    assert route_after_diagnose(state) == _END
