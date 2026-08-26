"""RobotPolicy Protocol — the sensorimotor boundary."""

from typing import Protocol, runtime_checkable

from langgraph_vla_agent.domain.actions import RobotAction
from langgraph_vla_agent.domain.context import PolicyContext
from langgraph_vla_agent.domain.observations import RobotObservation


@runtime_checkable
class RobotPolicy(Protocol):
    """Structural interface for all policy implementations.

    Implementations
    ---------------
    MockRobotPolicy     — deterministic, no deps (M1)
    ReplayRobotPolicy   — replays recorded actions (M2)
    SmolVLAPolicyAdapter — wraps lerobot/smolvla_base (M3)

    The Executor calls reset() once per episode and act() once per step.
    The policy never receives raw action commands from LangGraph — only the
    natural-language instruction and the current observation.
    """

    def reset(self, context: PolicyContext) -> None:
        """Prepare for a new episode.

        Called once before the first act() of each episode. Implementations
        may use context.seed for reproducibility and context.run_id for logging.
        """
        ...

    def act(self, observation: RobotObservation, instruction: str) -> RobotAction:
        """Produce an action given the current observation and instruction.

        Parameters
        ----------
        observation:
            Current sensor reading from the environment.
        instruction:
            Natural-language subtask description (e.g. "grasp the red cube").

        Returns
        -------
        RobotAction
            1-D action vector. The Executor validates finiteness and shape
            before passing it to the environment.
        """
        ...
