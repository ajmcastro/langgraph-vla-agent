"""MockEnvironment — scripted environment for testing the execution loop."""

from enum import StrEnum
from typing import Any

import numpy as np

from langgraph_vla_agent.domain.actions import RobotAction
from langgraph_vla_agent.domain.observations import RobotObservation
from langgraph_vla_agent.domain.results import StepResult
from langgraph_vla_agent.domain.tasks import SubTask


class MockScenario(StrEnum):
    """Controls when and how MockEnvironment signals episode termination."""

    SUCCEED_AT_STEP = "succeed_at_step"
    """Terminate with success=True at step N. Tests the SUCCESS path."""

    FAIL_AT_STEP = "fail_at_step"
    """Terminate with success=False (failure) at step N. Tests FAILURE path."""

    NEVER_TERMINATE = "never_terminate"
    """Never emit terminated or truncated. Tests MAX_STEPS_EXCEEDED path."""

    TRUNCATE_AT_STEP = "truncate_at_step"
    """Emit truncated=True at step N (not terminated). Tests the truncation
    branch, e.g. workspace limit exceeded."""


class MockEnvironment:
    """Scripted environment with configurable terminal conditions.

    Satisfies the RobotEnvironment Protocol without any simulator or dataset.

    Parameters
    ----------
    scenario:
        Which scripted terminal condition to use.
    n:
        Step at which the terminal condition fires (for scenarios that use it).
    state_dim:
        Dimension of the proprioceptive state vector in returned observations.
    """

    def __init__(
        self,
        scenario: MockScenario = MockScenario.SUCCEED_AT_STEP,
        n: int = 3,
        state_dim: int = 6,
    ) -> None:
        self.scenario = scenario
        self.n = n
        self.state_dim = state_dim
        self._step_count: int = 0
        self._current_subtask: SubTask | None = None

    def reset(self, subtask: SubTask) -> RobotObservation:
        self._step_count = 0
        self._current_subtask = subtask
        return self._make_observation()

    def step(self, action: RobotAction) -> tuple[RobotObservation, StepResult]:
        self._step_count += 1
        obs = self._make_observation()

        step_result = self._evaluate_terminal()
        return obs, step_result

    def observe(self) -> RobotObservation:
        return self._make_observation()

    @property
    def step_count(self) -> int:
        """Number of step() calls since the last reset()."""
        return self._step_count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_observation(self) -> RobotObservation:
        return RobotObservation(
            state=np.zeros(self.state_dim, dtype=np.float32),
            images={},
            timestamp=float(self._step_count),
        )

    def _evaluate_terminal(self) -> StepResult:
        info: dict[str, Any] = {"step": self._step_count, "scenario": self.scenario.value}

        if self.scenario == MockScenario.SUCCEED_AT_STEP and self._step_count >= self.n:
            return StepResult(terminated=True, truncated=False, success=True, info=info)

        if self.scenario == MockScenario.FAIL_AT_STEP and self._step_count >= self.n:
            return StepResult(terminated=True, truncated=False, success=False, info=info)

        if self.scenario == MockScenario.TRUNCATE_AT_STEP and self._step_count >= self.n:
            return StepResult(terminated=False, truncated=True, success=False, info=info)

        return StepResult(terminated=False, truncated=False, success=False, info=info)
