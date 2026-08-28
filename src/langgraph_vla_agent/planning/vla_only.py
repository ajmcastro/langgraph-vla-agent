"""VlaOnlyPlanner — passes the full goal to the policy as a single subtask.

This is the 'VLA-only' baseline condition in the M6 planning-granularity
experiment.  No decomposition is performed: the full goal text becomes the
policy instruction, exactly as in direct VLA execution.

By routing through the same AgentRunner infrastructure as the agentic
conditions, all three conditions produce identical metric shapes
(subtask counts, policy-call counts, retry/replan counts), making
comparison directly apples-to-apples.
"""

from langgraph_vla_agent.domain.tasks import SubTask, TaskGoal, TaskPlan


class VlaOnlyPlanner:
    """No-decomposition baseline: wraps the full goal into one subtask.

    Satisfies the TaskPlanner Protocol structurally — no inheritance required.
    """

    def plan(self, goal: TaskGoal) -> TaskPlan:
        """Return a single-subtask plan whose instruction is the full goal text."""
        return TaskPlan(
            goal=goal,
            subtasks=[SubTask(instruction=goal.text, success_criteria="task complete")],
            planner_id="vla_only",
        )
