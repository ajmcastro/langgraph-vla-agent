"""Unit tests for LLMTaskPlanner using a pure-Python stub LanguageModel."""

import json
from typing import Any

import pytest

from langgraph_vla_agent.domain.context import EvaluationMode
from langgraph_vla_agent.domain.tasks import TaskGoal
from langgraph_vla_agent.planning.base import PlanningError
from langgraph_vla_agent.planning.llm import LLMTaskPlanner

# ---------------------------------------------------------------------------
# Stub LanguageModel (no langchain import required)
# ---------------------------------------------------------------------------


class _FixedResponseLLM:
    """Minimal LanguageModel stub that returns a fixed JSON string."""

    def __init__(self, json_body: dict[str, Any] | None = None, *, raises: bool = False) -> None:
        self._body = json_body
        self._raises = raises

    def invoke(self, messages: list[Any]) -> Any:
        if self._raises:
            raise RuntimeError("simulated LLM failure")

        class _Msg:
            def __init__(self, content: str) -> None:
                self.content = content

        return _Msg(json.dumps(self._body))


def _goal(text: str) -> TaskGoal:
    return TaskGoal(text=text, run_id="t", evaluation_mode=EvaluationMode.MOCK)


_VALID_BODY = {
    "subtasks": [
        {"instruction": "pick up the object", "success_criteria": "grasped"},
        {"instruction": "place the object", "success_criteria": "placed"},
    ]
}

# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_parses_valid_json_response() -> None:
    planner = LLMTaskPlanner(_FixedResponseLLM(_VALID_BODY))
    plan = planner.plan(_goal("pick and place the cube"))
    assert len(plan.subtasks) == 2


def test_subtask_instructions_match_llm_response() -> None:
    planner = LLMTaskPlanner(_FixedResponseLLM(_VALID_BODY))
    plan = planner.plan(_goal("pick and place"))
    assert plan.subtasks[0].instruction == "pick up the object"
    assert plan.subtasks[1].instruction == "place the object"


def test_success_criteria_are_preserved() -> None:
    planner = LLMTaskPlanner(_FixedResponseLLM(_VALID_BODY))
    plan = planner.plan(_goal("pick and place"))
    assert plan.subtasks[0].success_criteria == "grasped"


def test_planner_id_is_set() -> None:
    planner = LLMTaskPlanner(_FixedResponseLLM(_VALID_BODY), planner_id="claude-sonnet")
    plan = planner.plan(_goal("pick and place"))
    assert plan.planner_id == "claude-sonnet"


def test_subtasks_have_unique_ids() -> None:
    planner = LLMTaskPlanner(_FixedResponseLLM(_VALID_BODY))
    plan = planner.plan(_goal("pick and place"))
    ids = [st.id for st in plan.subtasks]
    assert len(ids) == len(set(ids))


def test_missing_success_criteria_defaults_to_empty_string() -> None:
    body = {"subtasks": [{"instruction": "approach"}]}
    planner = LLMTaskPlanner(_FixedResponseLLM(body))
    plan = planner.plan(_goal("approach object"))
    assert plan.subtasks[0].success_criteria == ""


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_raises_on_llm_call_failure() -> None:
    planner = LLMTaskPlanner(_FixedResponseLLM(raises=True))
    with pytest.raises(PlanningError, match="LLM call failed"):
        planner.plan(_goal("pick and place"))


def test_raises_on_invalid_json() -> None:
    class _BadLLM:
        def invoke(self, messages: list[Any]) -> Any:
            class _Msg:
                content = "not valid json {"

            return _Msg()

    planner = LLMTaskPlanner(_BadLLM())
    with pytest.raises(PlanningError, match="invalid plan JSON"):
        planner.plan(_goal("pick and place"))


def test_raises_on_empty_subtask_list() -> None:
    planner = LLMTaskPlanner(_FixedResponseLLM({"subtasks": []}))
    with pytest.raises(PlanningError, match="empty subtask list"):
        planner.plan(_goal("pick and place"))


def test_raises_when_too_many_subtasks() -> None:
    body = {
        "subtasks": [{"instruction": f"step {i}"} for i in range(10)],
    }
    planner = LLMTaskPlanner(_FixedResponseLLM(body), max_subtasks=3)
    with pytest.raises(PlanningError, match="maximum allowed"):
        planner.plan(_goal("pick and place"))


def test_raises_when_subtasks_key_missing() -> None:
    planner = LLMTaskPlanner(_FixedResponseLLM({"steps": []}))
    with pytest.raises(PlanningError, match="invalid plan JSON"):
        planner.plan(_goal("pick and place"))
