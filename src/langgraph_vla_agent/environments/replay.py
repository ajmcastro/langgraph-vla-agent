"""ReplayEnvironment — replays recorded observations from a ReplayEpisode."""

import numpy as np

from langgraph_vla_agent.datasets.episode import ReplayEpisode, ReplayStep
from langgraph_vla_agent.domain.actions import RobotAction
from langgraph_vla_agent.domain.observations import RobotObservation
from langgraph_vla_agent.domain.results import StepResult
from langgraph_vla_agent.domain.tasks import SubTask


class ReplayEnvironment:
    """Satisfies RobotEnvironment Protocol by replaying recorded observations.

    On reset() it returns the observation of the first recorded step.
    On step() it returns the next recorded observation and the terminal
    signals (terminated, truncated, success) from the current step.

    Critical: the action argument to step() is accepted but IGNORED.
    This is the fundamental semantic of offline/replay evaluation: the
    environment serves the pre-recorded trajectory regardless of what action
    the policy or executor actually passes. It is therefore impossible to
    measure counterfactual behaviour (what would have happened if a
    different action had been taken).

    See docs/evaluation.md — "Offline / replay evaluation" for details on
    what replay evaluation can and cannot measure.

    Parameters
    ----------
    episode:
        The recorded episode whose observations will be replayed.
    """

    def __init__(self, episode: ReplayEpisode) -> None:
        self._episode = episode
        self._ptr: int = 0

    def reset(self, subtask: SubTask) -> RobotObservation:
        """Rewind to step 0 and return the first recorded observation."""
        self._ptr = 0
        return self._to_observation(self._episode.steps[0])

    def step(self, action: RobotAction) -> tuple[RobotObservation, StepResult]:
        """Advance the replay pointer and return the next observation + result.

        The ``action`` argument is accepted to satisfy the RobotEnvironment
        Protocol but is NOT used — the next observation comes from the
        recorded trajectory, not from simulating the action.

        If the pointer is already past the last step (episode over-run),
        the final step's observation is repeated and ``truncated=True`` is
        returned so the Executor exits cleanly.
        """
        if self._ptr >= len(self._episode.steps):
            last = self._episode.steps[-1]
            return self._to_observation(last), StepResult(
                terminated=False, truncated=True, success=False
            )

        current_step = self._episode.steps[self._ptr]
        self._ptr += 1

        next_obs = (
            self._to_observation(self._episode.steps[self._ptr])
            if self._ptr < len(self._episode.steps)
            else self._to_observation(current_step)
        )

        step_result = StepResult(
            terminated=current_step.terminated,
            truncated=current_step.truncated,
            success=current_step.success,
        )
        return next_obs, step_result

    def observe(self) -> RobotObservation:
        """Return the current observation without advancing the pointer."""
        idx = min(self._ptr, len(self._episode.steps) - 1)
        return self._to_observation(self._episode.steps[idx])

    @property
    def ptr(self) -> int:
        """Current step pointer (number of step() calls since last reset)."""
        return self._ptr

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_observation(self, step: ReplayStep) -> RobotObservation:
        state = np.array(step.observation["state"], dtype=np.float32)
        return RobotObservation(state=state, timestamp=float(step.timestep))
