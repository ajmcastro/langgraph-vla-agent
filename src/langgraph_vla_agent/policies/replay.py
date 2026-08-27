"""ReplayRobotPolicy — replays recorded actions from a ReplayEpisode."""

import numpy as np

from langgraph_vla_agent.datasets.episode import ReplayEpisode
from langgraph_vla_agent.domain.actions import RobotAction
from langgraph_vla_agent.domain.context import PolicyContext
from langgraph_vla_agent.domain.observations import RobotObservation


class ReplayRobotPolicy:
    """Satisfies RobotPolicy Protocol by replaying recorded actions in order.

    On each act() call the next recorded action from the episode is returned.
    On reset() the step pointer is rewound to zero, making the policy
    reusable across multiple independent Executor.run() calls on the same
    instance.

    This is the policy used in offline/replay evaluation (EvaluationMode.REPLAY).
    It does NOT call any model — it is a data-playback device.

    Important limitation
    --------------------
    The observation argument to act() is accepted but ignored. In replay
    mode, the recorded action sequence is served regardless of what the
    environment actually observes. This is the key constraint of offline
    evaluation: the policy cannot respond to deviations from the recorded
    trajectory.

    Parameters
    ----------
    episode:
        The recorded episode whose actions will be replayed.
    """

    def __init__(self, episode: ReplayEpisode) -> None:
        self._episode = episode
        self._ptr: int = 0

    def reset(self, context: PolicyContext) -> None:
        """Rewind to step 0 for a fresh episode."""
        self._ptr = 0

    def act(self, observation: RobotObservation, instruction: str) -> RobotAction:
        """Return the next recorded action.

        Parameters
        ----------
        observation:
            Ignored — replay serves the pre-recorded action regardless.
        instruction:
            Ignored — the recorded instruction is embedded in the episode.

        Raises
        ------
        IndexError
            If act() is called more times than there are recorded steps.
            This should not happen in normal usage because the ReplayEnvironment
            signals termination on the last step before the policy is exhausted.
        """
        if self._ptr >= len(self._episode.steps):
            raise IndexError(
                f"ReplayRobotPolicy: episode {self._episode.episode_id!r} "
                f"exhausted after {len(self._episode.steps)} steps "
                f"(ptr={self._ptr})"
            )
        action_values = np.array(self._episode.steps[self._ptr].action, dtype=np.float32)
        self._ptr += 1
        return RobotAction(values=action_values)

    @property
    def ptr(self) -> int:
        """Current step pointer (number of act() calls since last reset)."""
        return self._ptr
