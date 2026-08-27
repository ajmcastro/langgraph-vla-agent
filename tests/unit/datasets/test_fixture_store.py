"""Unit tests for FixtureEpisodeStore — reads committed JSON fixtures."""

import pathlib

import pytest

from langgraph_vla_agent.datasets.episode import ReplayEpisode
from langgraph_vla_agent.datasets.store import EpisodeStore, FixtureEpisodeStore

# Resolve the fixture directory from this file's location (project-root-relative).
_FIXTURES_DIR = pathlib.Path(__file__).parents[3] / "data" / "fixtures" / "episodes"


def _store() -> FixtureEpisodeStore:
    return FixtureEpisodeStore(_FIXTURES_DIR)


# ---------------------------------------------------------------------------
# list_episodes
# ---------------------------------------------------------------------------


def test_list_episodes_returns_three_fixtures() -> None:
    episodes = _store().list_episodes()
    assert len(episodes) == 3


def test_list_episodes_is_sorted() -> None:
    episodes = _store().list_episodes()
    assert episodes == sorted(episodes)


def test_list_episodes_returns_stems_not_paths() -> None:
    episodes = _store().list_episodes()
    for ep_id in episodes:
        assert not ep_id.endswith(".json")


# ---------------------------------------------------------------------------
# load_episode
# ---------------------------------------------------------------------------


def test_load_episode_returns_replay_episode() -> None:
    ep = _store().load_episode("fixture_episode_001")
    assert isinstance(ep, ReplayEpisode)
    assert ep.episode_id == "fixture_episode_001"


def test_load_episode_has_correct_action_dim() -> None:
    ep = _store().load_episode("fixture_episode_001")
    assert ep.action_dim == 6
    for step in ep.steps:
        assert len(step.action) == 6


def test_load_episode_success_on_last_step() -> None:
    ep = _store().load_episode("fixture_episode_001")
    assert ep.steps[-1].terminated is True
    assert ep.steps[-1].success is True


def test_load_episode_failure_episode() -> None:
    ep = _store().load_episode("fixture_episode_002")
    assert ep.steps[-1].terminated is True
    assert ep.steps[-1].success is False


def test_load_episode_not_found_raises() -> None:
    with pytest.raises(FileNotFoundError, match="does_not_exist"):
        _store().load_episode("does_not_exist")


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_fixture_store_satisfies_protocol() -> None:
    store = _store()
    assert isinstance(store, EpisodeStore)
