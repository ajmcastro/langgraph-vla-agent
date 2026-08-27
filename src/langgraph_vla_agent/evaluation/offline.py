"""OfflineEvaluator — action prediction error across recorded episodes."""

import math

import numpy as np
import structlog

from langgraph_vla_agent.datasets.episode import ReplayEpisode
from langgraph_vla_agent.datasets.store import EpisodeStore
from langgraph_vla_agent.domain.context import EvaluationMode, PolicyContext
from langgraph_vla_agent.domain.tasks import SubTask
from langgraph_vla_agent.environments.replay import ReplayEnvironment
from langgraph_vla_agent.evaluation.metrics import (
    ActionErrorMetrics,
    EpisodeEvalResult,
    OfflineEvalResult,
)
from langgraph_vla_agent.policies.base import RobotPolicy

_log = structlog.get_logger(__name__)


class OfflineEvaluator:
    """Measures action prediction error of a policy against recorded episodes.

    For each step in each episode the evaluator:
      1. Passes the recorded observation to the policy.
      2. Compares the predicted action to the recorded ground-truth action.
      3. Computes per-step L1 and L2 error across action dimensions.

    The replay environment is used to advance through recorded observations;
    the action passed to env.step() is the POLICY'S prediction (the replay
    environment ignores it and serves the next recorded observation anyway).

    Critical limitation
    -------------------
    Action prediction error is a necessary but not sufficient condition for
    task success. Two policies with identical action error can have very
    different closed-loop performance because the replay environment cannot
    simulate counterfactual trajectories. Every OfflineEvalResult carries
    this note in its ``evaluation_note`` field.

    Parameters
    ----------
    policy:
        Any RobotPolicy implementation (ReplayRobotPolicy, SmolVLAPolicyAdapter, etc.).
    store:
        Episode source (FixtureEpisodeStore for unit tests; HubEpisodeStore for M4+).
    model_id:
        Human-readable model identifier recorded in the result.
    dataset_id:
        Human-readable dataset identifier recorded in the result.
    evaluation_mode:
        Must be REPLAY for offline evaluation.
    """

    def __init__(
        self,
        policy: RobotPolicy,
        store: EpisodeStore,
        *,
        model_id: str = "unknown",
        dataset_id: str = "fixture",
        evaluation_mode: EvaluationMode = EvaluationMode.REPLAY,
    ) -> None:
        self._policy = policy
        self._store = store
        self._model_id = model_id
        self._dataset_id = dataset_id
        self._evaluation_mode = evaluation_mode

    def evaluate(
        self,
        episode_ids: list[str],
        *,
        run_id: str = "offline-eval",
    ) -> OfflineEvalResult:
        """Evaluate the policy on the given episodes and return aggregate metrics.

        Parameters
        ----------
        episode_ids:
            Episode IDs to load from the store and evaluate. All must exist.
        run_id:
            Identifier for this evaluation run, used in log fields.

        Returns
        -------
        OfflineEvalResult
            Aggregate and per-episode action error metrics.
        """
        log = _log.bind(
            run_id=run_id,
            model_id=self._model_id,
            dataset_id=self._dataset_id,
            n_episodes=len(episode_ids),
        )
        log.info("offline_evaluator.start")

        per_episode: list[EpisodeEvalResult] = []
        for i, ep_id in enumerate(episode_ids):
            episode = self._store.load_episode(ep_id)
            context = PolicyContext(
                run_id=run_id,
                episode_id=ep_id,
                evaluation_mode=self._evaluation_mode,
            )
            ep_result = self._evaluate_episode(episode, context)
            per_episode.append(ep_result)
            log.debug(
                "offline_evaluator.episode_done",
                episode_id=ep_id,
                n_steps=ep_result.n_steps,
                l1_mean=round(ep_result.l1_mean, 6),
                i=i + 1,
            )

        aggregate = self._aggregate(per_episode)
        log.info(
            "offline_evaluator.done",
            l1_mean=round(aggregate.l1_mean, 6),
            l2_mean=round(aggregate.l2_mean, 6),
            n_steps=aggregate.n_steps,
        )
        return OfflineEvalResult(
            evaluation_mode=self._evaluation_mode,
            model_id=self._model_id,
            dataset_id=self._dataset_id,
            n_episodes=len(per_episode),
            aggregate=aggregate,
            per_episode=per_episode,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_episode(
        self,
        episode: ReplayEpisode,
        context: PolicyContext,
    ) -> EpisodeEvalResult:
        env = ReplayEnvironment(episode)
        subtask = SubTask(id=episode.episode_id, instruction=episode.instruction)

        self._policy.reset(context)
        obs = env.reset(subtask)

        l1_errors: list[float] = []
        l2_errors: list[float] = []

        for step in episode.steps:
            predicted = self._policy.act(obs, episode.instruction)
            gt_action = np.array(step.action, dtype=np.float32)

            diff = predicted.values - gt_action
            l1_errors.append(float(np.mean(np.abs(diff))))
            l2_errors.append(float(np.sqrt(np.mean(diff**2))))

            if step.terminated or step.truncated:
                break
            obs, _ = env.step(predicted)

        return EpisodeEvalResult(
            episode_id=episode.episode_id,
            instruction=episode.instruction,
            n_steps=len(l1_errors),
            action_errors_l1=l1_errors,
            action_errors_l2=l2_errors,
        )

    @staticmethod
    def _aggregate(results: list[EpisodeEvalResult]) -> ActionErrorMetrics:
        all_l1 = [e for r in results for e in r.action_errors_l1]
        all_l2 = [e for r in results for e in r.action_errors_l2]

        if not all_l1:
            return ActionErrorMetrics(l1_mean=0.0, l1_std=0.0, l2_mean=0.0, l2_std=0.0, n_steps=0)

        def _mean(xs: list[float]) -> float:
            return sum(xs) / len(xs)

        def _std(xs: list[float]) -> float:
            m = _mean(xs)
            return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

        return ActionErrorMetrics(
            l1_mean=_mean(all_l1),
            l1_std=_std(all_l1),
            l2_mean=_mean(all_l2),
            l2_std=_std(all_l2),
            n_steps=len(all_l1),
        )
