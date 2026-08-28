"""LLMTaskPlanner — structured LLM-backed task decomposition."""

import json
from typing import Any

import structlog
from pydantic import BaseModel, ValidationError

from langgraph_vla_agent.domain.tasks import SubTask, TaskGoal, TaskPlan
from langgraph_vla_agent.planning.base import LanguageModel, PlanningError

_log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a robot task planner. Decompose a high-level manipulation goal into an \
ordered sequence of subtasks that a robot arm with a gripper can execute.

Rules:
- Each subtask must be a high-level physical action: approach, grasp, lift, move, place, push, pull.
- Never generate servo positions, joint torques, or low-level motor commands.
- Use between 2 and 6 subtasks. Prefer fewer, meaningful steps.
- Return ONLY a valid JSON object. No markdown, no explanation, no code fences.

Required JSON schema:
{
  "subtasks": [
    {"instruction": "<action description>", "success_criteria": "<observable outcome>"},
    ...
  ]
}
"""

# ---------------------------------------------------------------------------
# Internal response model (what the LLM must return)
# ---------------------------------------------------------------------------


class _SubTaskSpec(BaseModel):
    instruction: str
    success_criteria: str = ""


class _PlanResponse(BaseModel):
    subtasks: list[_SubTaskSpec]


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class LLMTaskPlanner:
    """Planner that uses an LLM to decompose a natural-language goal into subtasks.

    The model is injected as a LanguageModel Protocol, so any langchain_core
    BaseChatModel (Claude, GPT-4, etc.) or a plain Python stub works.

    Provider packages (langchain-anthropic, langchain-openai, …) are NOT
    project dependencies — the user installs the one they need and passes
    the model instance here.

    Parameters
    ----------
    model:
        Any object satisfying the LanguageModel Protocol.
    planner_id:
        Identifier embedded in the resulting TaskPlan for tracing.
    max_subtasks:
        Upper bound enforced after parsing; prevents runaway plans.
    """

    def __init__(
        self,
        model: LanguageModel,
        planner_id: str = "llm",
        max_subtasks: int = 6,
    ) -> None:
        self._model = model
        self._planner_id = planner_id
        self._max_subtasks = max_subtasks

    def plan(self, goal: TaskGoal) -> TaskPlan:
        """Call the LLM and parse its response into a TaskPlan.

        Raises
        ------
        PlanningError
            On LLM call failure, invalid JSON, schema mismatch, empty list,
            or exceeding max_subtasks.
        """
        messages: list[Any] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Decompose this goal into subtasks:\n{goal.text}"},
        ]

        log = _log.bind(run_id=goal.run_id, planner_id=self._planner_id)
        log.info("llm_planner.calling_model", goal=goal.text)

        try:
            response = self._model.invoke(messages)
        except Exception as exc:
            raise PlanningError(f"LLM call failed: {exc}") from exc

        # langchain AIMessage has .content; plain strings work too.
        raw: str = response.content if hasattr(response, "content") else str(response)

        log.debug("llm_planner.raw_response", raw=raw[:500])

        try:
            data = json.loads(raw)
            plan_response = _PlanResponse.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise PlanningError(
                f"LLM returned invalid plan JSON: {exc}\nRaw response: {raw[:300]!r}"
            ) from exc

        if not plan_response.subtasks:
            raise PlanningError("LLM returned an empty subtask list")

        if len(plan_response.subtasks) > self._max_subtasks:
            raise PlanningError(
                f"LLM returned {len(plan_response.subtasks)} subtasks; "
                f"maximum allowed is {self._max_subtasks}"
            )

        subtasks = [
            SubTask(instruction=s.instruction, success_criteria=s.success_criteria)
            for s in plan_response.subtasks
        ]
        log.info("llm_planner.plan_created", n_subtasks=len(subtasks))

        return TaskPlan(
            goal=goal,
            subtasks=subtasks,
            planner_id=self._planner_id,
        )
