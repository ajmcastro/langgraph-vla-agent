"""DeterministicPlanner — rule-based, no LLM, reproducible task decomposition."""

from typing import Literal

from langgraph_vla_agent.domain.tasks import SubTask, TaskGoal, TaskPlan
from langgraph_vla_agent.planning.base import PlanningError

# ---------------------------------------------------------------------------
# Task templates
# ---------------------------------------------------------------------------

# Keywords that trigger each task type (checked against lowercased goal text).
_PICK_PLACE_KEYWORDS = frozenset(["pick", "grasp", "grab", "take", "place", "put", "move", "set"])

# Coarse decomposition: 2 subtasks — one for pick, one for place.
# Appropriate for M6 "coarse agentic" condition.
_COARSE_SUBTASKS = [
    ("pick up the object", "object grasped and lifted clear of surface"),
    ("place the object at the target", "object released at target position"),
]

# Fine decomposition: 5 subtasks — full manipulation skill sequence.
# Appropriate for M6 "fine agentic" condition.
_FINE_SUBTASKS = [
    ("approach the object", "gripper positioned above the object"),
    ("grasp the object", "gripper closed around the object"),
    ("lift the object", "object lifted clear of surface"),
    ("move to target position", "gripper positioned above target zone"),
    ("place the object", "object released at target position"),
]


class DeterministicPlanner:
    """Rule-based planner that matches goals to predefined task templates.

    No LLM or network call is made. Plans are reproducible across runs.
    Supports two granularity levels for the M6 planning-granularity experiment.

    Parameters
    ----------
    granularity:
        "coarse" produces 2 broad subtasks; "fine" produces 5 detailed skills.
    """

    def __init__(self, granularity: Literal["coarse", "fine"] = "coarse") -> None:
        self._granularity = granularity

    @property
    def granularity(self) -> str:
        return self._granularity

    def plan(self, goal: TaskGoal) -> TaskPlan:
        """Match the goal to a template and return a TaskPlan.

        Raises
        ------
        PlanningError
            If the goal text contains no recognised pick-and-place keywords.
        """
        template = self._match_template(goal.text)
        subtasks = self._instantiate(template, goal)
        return TaskPlan(
            goal=goal,
            subtasks=subtasks,
            planner_id=f"deterministic/{self._granularity}",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _match_template(self, text: str) -> str:
        lower = text.lower()
        if any(kw in lower for kw in _PICK_PLACE_KEYWORDS):
            return "pick_and_place"
        raise PlanningError(
            f"No deterministic template matches goal: {text!r}. "
            f"Known keywords: {sorted(_PICK_PLACE_KEYWORDS)}"
        )

    def _instantiate(self, template: str, goal: TaskGoal) -> list[SubTask]:
        if template != "pick_and_place":
            raise PlanningError(f"Unknown template: {template!r}")

        rows = _COARSE_SUBTASKS if self._granularity == "coarse" else _FINE_SUBTASKS
        return [SubTask(instruction=instr, success_criteria=criteria) for instr, criteria in rows]
