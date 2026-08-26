"""RobotEnvironment Protocol — the sensorimotor world boundary."""

from typing import Protocol, runtime_checkable

from langgraph_vla_agent.domain.actions import RobotAction
from langgraph_vla_agent.domain.observations import RobotObservation
from langgraph_vla_agent.domain.results import StepResult
from langgraph_vla_agent.domain.tasks import SubTask


@runtime_checkable
class RobotEnvironment(Protocol):
    """Structural interface for all environment implementations.

    Implementations
    ---------------
    MockEnvironment        — scripted scenarios, no deps (M1)
    ReplayEnvironment      — replays recorded episodes (M2)
    SimulationEnvironment  — physics simulator (M7, optional)
    HardwareEnvironment    — physical robot (future, isolated)

    The environment is responsible for:
    - Returning the initial observation after reset().
    - Applying actions and returning the next observation + step result.
    - Signalling termination (terminated) and truncation (truncated).
    - Evaluating the success predicate (StepResult.success).
    """

    def reset(self, subtask: SubTask) -> RobotObservation:
        """Initialise the environment for a new subtask episode.

        Returns the initial observation. May use subtask.instruction to select
        a relevant starting scene in replay or simulation backends.
        """
        ...

    def step(self, action: RobotAction) -> tuple[RobotObservation, StepResult]:
        """Apply action, advance world state, return next obs and step result.

        The Executor validates the action before calling step(). Implementations
        must not crash on out-of-range but valid-shape actions — return a
        StepResult with terminated=True and success=False instead.
        """
        ...

    def observe(self) -> RobotObservation:
        """Return the current observation without advancing the world state.

        Used by the Executor on the first step before any action is applied,
        and can be called by diagnostics without side effects.
        """
        ...
