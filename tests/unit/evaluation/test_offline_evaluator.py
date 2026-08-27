"""Unit tests for OfflineEvaluator."""

import math
from pathlib import Path

import pytest

from langgraph_vla_agent.datasets.episode import ReplayEpisode, ReplayStep
from langgraph_vla_agent.datasets.store import FixtureEpisodeStore
from langgraph_vla_agent.domain.context import EvaluationMode
from langgraph_vla_agent.evaluation import OfflineEvaluator
from langgraph_vla_agent.policies import ReplayRobotPolicy
from langgraph_vla_agent.policies.smolvla import SmolVLAPolicyAdapter, _StubSmolVLAModel

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_FIXTURE_DIR = Path(__file__).parents[3] / "data" / "fixtures" / "episodes"


def _make_episode(actions: list[list[float]], episode_id: str = "ep-t") -> ReplayEpisode:
    steps = [
        ReplayStep(
            timestep=i,
            observation={"state": [0.0] * len(a)},
            action=a,
            terminated=(i == len(actions) - 1),
            success=(i == len(actions) - 1),
        )
        for i, a in enumerate(actions)
    ]
    return ReplayEpisode(
        episode_id=episode_id,
        instruction="pick the block",
        dataset_id="fixture",
        action_dim=len(actions[0]),
        state_dim=len(actions[0]),
        steps=steps,
    )


class _InMemoryStore:
    """Minimal in-memory EpisodeStore for testing."""

    def __init__(self, episodes: list[ReplayEpisode]) -> None:
        self._eps = {ep.episode_id: ep for ep in episodes}

    def list_episodes(self) -> list[str]:
        return sorted(self._eps)

    def load_episode(self, episode_id: str) -> ReplayEpisode:
        return self._eps[episode_id]


# ---------------------------------------------------------------------------
# Zero-error baseline: ReplayRobotPolicy predicts exact recorded actions
# ---------------------------------------------------------------------------


def test_offline_evaluator_zero_error_with_replay_policy() -> None:
    actions = [[0.1, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.2, 0.0, 0.0, 0.0, 0.0]]
    ep = _make_episode(actions)
    store = _InMemoryStore([ep])
    policy = ReplayRobotPolicy(ep)
    evaluator = OfflineEvaluator(
        policy,
        store,
        model_id="replay-baseline",
        dataset_id="fixture",
        evaluation_mode=EvaluationMode.REPLAY,
    )
    result = evaluator.evaluate([ep.episode_id])
    assert result.aggregate.l1_mean == pytest.approx(0.0)
    assert result.aggregate.l2_mean == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Non-zero error: stub always returns zeros, fixtures have non-zero actions
# ---------------------------------------------------------------------------


def test_offline_evaluator_nonzero_error_with_stub() -> None:
    actions = [[0.1, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.2, 0.0, 0.0, 0.0, 0.0]]
    ep = _make_episode(actions)
    store = _InMemoryStore([ep])
    stub = _StubSmolVLAModel(action_dim=6)
    policy = SmolVLAPolicyAdapter(_model=stub)
    evaluator = OfflineEvaluator(policy, store, model_id="stub", dataset_id="fixture")
    result = evaluator.evaluate([ep.episode_id])
    assert result.aggregate.l1_mean > 0.0


# ---------------------------------------------------------------------------
# n_steps counts correct number of steps per episode
# ---------------------------------------------------------------------------


def test_offline_evaluator_counts_steps_correctly() -> None:
    actions = [[float(i)] + [0.0] * 5 for i in range(3)]
    ep = _make_episode(actions)
    store = _InMemoryStore([ep])
    policy = ReplayRobotPolicy(ep)
    evaluator = OfflineEvaluator(policy, store)
    result = evaluator.evaluate([ep.episode_id])
    assert result.per_episode[0].n_steps == 3
    assert result.aggregate.n_steps == 3


# ---------------------------------------------------------------------------
# Multiple episodes: aggregate covers all steps
# ---------------------------------------------------------------------------


def test_offline_evaluator_aggregates_multiple_episodes() -> None:
    ep1 = _make_episode([[0.1] * 6, [0.2] * 6], episode_id="ep-a")
    ep2 = _make_episode([[0.3] * 6, [0.4] * 6, [0.5] * 6], episode_id="ep-b")
    store = _InMemoryStore([ep1, ep2])

    class _SwitchingReplayPolicy:
        """Acts as replay policy for whichever episode was last reset."""

        def __init__(self) -> None:
            self._inner: ReplayRobotPolicy | None = None

        def reset(self, context):  # type: ignore[override]
            ep = store.load_episode(context.episode_id)
            self._inner = ReplayRobotPolicy(ep)
            self._inner.reset(context)

        def act(self, obs, instruction):  # type: ignore[override]
            assert self._inner is not None
            return self._inner.act(obs, instruction)

    evaluator = OfflineEvaluator(_SwitchingReplayPolicy(), store)  # type: ignore[arg-type]
    result = evaluator.evaluate(["ep-a", "ep-b"])
    assert result.n_episodes == 2
    assert result.aggregate.n_steps == 5  # 2 + 3
    assert result.aggregate.l1_mean == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Evaluation note is always present
# ---------------------------------------------------------------------------


def test_offline_eval_result_always_has_evaluation_note() -> None:
    ep = _make_episode([[0.0] * 6])
    store = _InMemoryStore([ep])
    policy = ReplayRobotPolicy(ep)
    evaluator = OfflineEvaluator(policy, store)
    result = evaluator.evaluate([ep.episode_id])
    assert result.evaluation_note
    assert "closed-loop" in result.evaluation_note


# ---------------------------------------------------------------------------
# Empty episode list: returns zero-step aggregate
# ---------------------------------------------------------------------------


def test_offline_evaluator_empty_episode_list() -> None:
    ep = _make_episode([[0.1] * 6])
    store = _InMemoryStore([ep])
    policy = ReplayRobotPolicy(ep)
    evaluator = OfflineEvaluator(policy, store)
    result = evaluator.evaluate([])
    assert result.n_episodes == 0
    assert result.aggregate.n_steps == 0


# ---------------------------------------------------------------------------
# Fixture store integration: uses the real fixture JSON files
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _FIXTURE_DIR.exists(),
    reason="Fixture directory not found",
)
def test_offline_evaluator_with_fixture_store_zero_error() -> None:
    store = FixtureEpisodeStore(_FIXTURE_DIR)
    ep_ids = store.list_episodes()
    assert len(ep_ids) >= 1

    ep = store.load_episode(ep_ids[0])
    policy = ReplayRobotPolicy(ep)
    evaluator = OfflineEvaluator(
        policy,
        store,
        model_id="replay-baseline",
        dataset_id="fixture",
    )
    result = evaluator.evaluate([ep_ids[0]])
    assert result.aggregate.l1_mean == pytest.approx(0.0)
    assert result.aggregate.n_steps == ep.length


# ---------------------------------------------------------------------------
# L1 / L2 error values match manual calculation
# ---------------------------------------------------------------------------


def test_l1_l2_error_values_are_correct() -> None:
    # One step: predicted=zeros(3), gt=[1,2,3]
    # L1 = mean(|0-1|, |0-2|, |0-3|) = mean(1,2,3) = 2.0
    # L2 = sqrt(mean(1,4,9)) = sqrt(14/3) ≈ 2.1602
    ep = _make_episode([[1.0, 2.0, 3.0]])
    store = _InMemoryStore([ep])
    stub = _StubSmolVLAModel(action_dim=3)
    policy = SmolVLAPolicyAdapter(_model=stub)
    evaluator = OfflineEvaluator(policy, store)
    result = evaluator.evaluate([ep.episode_id])
    assert result.aggregate.l1_mean == pytest.approx(2.0)
    expected_l2 = math.sqrt(14.0 / 3.0)
    assert result.aggregate.l2_mean == pytest.approx(expected_l2, rel=1e-4)
