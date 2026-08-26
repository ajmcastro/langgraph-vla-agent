"""MockRobotPolicy — deterministic policy for testing the execution loop."""

from enum import StrEnum

import numpy as np

from langgraph_vla_agent.domain.actions import RobotAction
from langgraph_vla_agent.domain.context import PolicyContext
from langgraph_vla_agent.domain.observations import RobotObservation


class MockPolicyBehavior(StrEnum):
    """Controls what MockRobotPolicy returns at each step."""

    ALWAYS_VALID = "always_valid"
    """Return a zero action vector every step. Never signals failure."""

    INVALID_AFTER_N = "invalid_after_n"
    """Return a valid action for steps 1..N, then NaN values to trigger the
    Executor's action-validation gate. Tests the INVALID_ACTION path."""

    RAISE_AFTER_N = "raise_after_n"
    """Raise RuntimeError at step N+1 to test the POLICY_ERROR path."""


class MockRobotPolicy:
    """Deterministic policy with configurable behaviour for unit tests.

    Satisfies the RobotPolicy Protocol without any ML dependencies.
    All state is reset by reset() to keep tests independent.

    Parameters
    ----------
    behavior:
        Which scripted scenario to execute.
    action_dim:
        Dimensionality of the returned action vector.
    n:
        Step threshold for INVALID_AFTER_N and RAISE_AFTER_N behaviors.
        The policy behaves normally for steps 1..n, then triggers at step n+1.
    """

    def __init__(
        self,
        behavior: MockPolicyBehavior = MockPolicyBehavior.ALWAYS_VALID,
        action_dim: int = 6,
        n: int = 3,
    ) -> None:
        self.behavior = behavior
        self.action_dim = action_dim
        self.n = n
        self._step_count: int = 0
        self._last_context: PolicyContext | None = None

    def reset(self, context: PolicyContext) -> None:
        self._step_count = 0
        self._last_context = context

    def act(self, observation: RobotObservation, instruction: str) -> RobotAction:
        self._step_count += 1

        if self.behavior == MockPolicyBehavior.INVALID_AFTER_N and self._step_count > self.n:
            return RobotAction(values=np.full(self.action_dim, np.nan, dtype=np.float32))

        if self.behavior == MockPolicyBehavior.RAISE_AFTER_N and self._step_count > self.n:
            raise RuntimeError(
                f"MockRobotPolicy: simulated policy crash at step {self._step_count}"
            )

        return RobotAction(values=np.zeros(self.action_dim, dtype=np.float32))

    @property
    def step_count(self) -> int:
        """Number of act() calls since the last reset()."""
        return self._step_count

    @property
    def last_context(self) -> PolicyContext | None:
        """The PolicyContext received by the most recent reset() call."""
        return self._last_context
