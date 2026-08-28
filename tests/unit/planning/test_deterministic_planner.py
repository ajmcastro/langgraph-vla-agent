"""Unit tests for DeterministicPlanner."""

import pytest

from langgraph_vla_agent.domain.context import EvaluationMode
from langgraph_vla_agent.domain.tasks import TaskGoal
from langgraph_vla_agent.planning.base import PlanningError
from langgraph_vla_agent.planning.deterministic import DeterministicPlanner


def _goal(text: str) -> TaskGoal:
    return TaskGoal(text=text, run_id="test-run", evaluation_mode=EvaluationMode.MOCK)


# ---------------------------------------------------------------------------
# Coarse granularity
# ---------------------------------------------------------------------------


def test_coarse_pick_and_place_returns_two_subtasks() -> None:
    planner = DeterministicPlanner(granularity="coarse")
    plan = planner.plan(_goal("pick up the cube and place it in the bin"))
    assert len(plan.subtasks) == 2


def test_coarse_planner_id_matches_granularity() -> None:
    planner = DeterministicPlanner(granularity="coarse")
    plan = planner.plan(_goal("pick up the cube"))
    assert plan.planner_id == "deterministic/coarse"


def test_coarse_subtask_instructions_are_non_empty() -> None:
    planner = DeterministicPlanner(granularity="coarse")
    plan = planner.plan(_goal("pick up the block"))
    for st in plan.subtasks:
        assert st.instruction.strip()


def test_coarse_subtasks_have_success_criteria() -> None:
    planner = DeterministicPlanner(granularity="coarse")
    plan = planner.plan(_goal("grab the object and set it down"))
    for st in plan.subtasks:
        assert st.success_criteria.strip()


# ---------------------------------------------------------------------------
# Fine granularity
# ---------------------------------------------------------------------------


def test_fine_pick_and_place_returns_five_subtasks() -> None:
    planner = DeterministicPlanner(granularity="fine")
    plan = planner.plan(_goal("pick the cube and place it"))
    assert len(plan.subtasks) == 5


def test_fine_planner_id_matches_granularity() -> None:
    planner = DeterministicPlanner(granularity="fine")
    plan = planner.plan(_goal("move the block to the target"))
    assert plan.planner_id == "deterministic/fine"


def test_fine_subtask_instructions_are_unique() -> None:
    planner = DeterministicPlanner(granularity="fine")
    plan = planner.plan(_goal("pick up the block"))
    instructions = [st.instruction for st in plan.subtasks]
    assert len(instructions) == len(set(instructions)), "subtask instructions must be distinct"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_unknown_goal_raises_planning_error() -> None:
    planner = DeterministicPlanner()
    with pytest.raises(PlanningError):
        planner.plan(_goal("dance around the room"))


def test_empty_goal_raises_planning_error() -> None:
    planner = DeterministicPlanner()
    with pytest.raises(PlanningError):
        planner.plan(_goal(""))


# ---------------------------------------------------------------------------
# TaskPlan structure
# ---------------------------------------------------------------------------


def test_plan_goal_is_preserved() -> None:
    planner = DeterministicPlanner()
    goal = _goal("pick up the cube")
    plan = planner.plan(goal)
    assert plan.goal is goal


def test_plan_version_defaults_to_zero() -> None:
    planner = DeterministicPlanner()
    plan = planner.plan(_goal("pick up the cube"))
    assert plan.plan_version == 0


def test_subtasks_have_auto_generated_unique_ids() -> None:
    planner = DeterministicPlanner(granularity="fine")
    plan = planner.plan(_goal("grab the block"))
    ids = [st.id for st in plan.subtasks]
    assert len(ids) == len(set(ids))


def test_plan_pending_excludes_completed() -> None:
    planner = DeterministicPlanner(granularity="coarse")
    plan = planner.plan(_goal("pick the block"))
    first_id = plan.subtasks[0].id
    pending = plan.pending_subtasks({first_id}, set())
    assert len(pending) == 1
    assert pending[0].id != first_id


def test_plan_is_complete_when_all_ids_present() -> None:
    planner = DeterministicPlanner(granularity="coarse")
    plan = planner.plan(_goal("pick the block"))
    all_ids = {st.id for st in plan.subtasks}
    assert plan.is_complete(all_ids)


def test_plan_is_not_complete_when_ids_missing() -> None:
    planner = DeterministicPlanner(granularity="fine")
    plan = planner.plan(_goal("pick the block"))
    partial = {plan.subtasks[0].id}
    assert not plan.is_complete(partial)
