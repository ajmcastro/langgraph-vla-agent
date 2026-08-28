"""Planning abstractions: PlanningError, TaskPlanner Protocol, LanguageModel Protocol."""

from typing import Protocol, runtime_checkable

from langgraph_vla_agent.domain.tasks import TaskGoal, TaskPlan


class PlanningError(Exception):
    """Raised when a planner cannot produce a valid plan from a goal."""


@runtime_checkable
class TaskPlanner(Protocol):
    """Structural interface for all planners.

    Both DeterministicPlanner and LLMTaskPlanner satisfy this Protocol.
    Tests can inject any object with a compatible plan() method.
    """

    def plan(self, goal: TaskGoal) -> TaskPlan:
        """Decompose a TaskGoal into an ordered TaskPlan.

        Raises
        ------
        PlanningError
            If the goal cannot be decomposed (unknown template, LLM failure,
            schema validation error, empty subtask list, etc.).
        """
        ...


@runtime_checkable
class LanguageModel(Protocol):
    """Minimal interface required by LLMTaskPlanner.

    Any langchain_core.BaseChatModel satisfies this Protocol structurally.
    A plain Python stub can be used in unit tests without installing langchain.

    The invoke() method receives a list of message dicts ({"role": ..., "content": ...})
    and returns an object with a .content attribute (or a plain string).
    """

    def invoke(self, messages: list[object]) -> object:
        """Call the model synchronously and return the response."""
        ...
