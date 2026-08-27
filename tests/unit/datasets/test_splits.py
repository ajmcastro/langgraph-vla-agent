"""Unit tests for EpisodeSplitter — deterministic, leak-free splits."""

import pytest

from langgraph_vla_agent.datasets.splits import EpisodeSplitter

_EPISODE_IDS = [f"ep-{i:02d}" for i in range(20)]


def _splitter() -> EpisodeSplitter:
    return EpisodeSplitter()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_split_is_deterministic_with_same_seed() -> None:
    s1 = _splitter().split(_EPISODE_IDS, seed=42)
    s2 = _splitter().split(_EPISODE_IDS, seed=42)
    assert s1.train == s2.train
    assert s1.val == s2.val
    assert s1.test == s2.test


def test_split_differs_with_different_seeds() -> None:
    s1 = _splitter().split(_EPISODE_IDS, seed=1)
    s2 = _splitter().split(_EPISODE_IDS, seed=2)
    # With 20 episodes, two seeds almost certainly produce different orderings
    assert s1.train != s2.train or s1.val != s2.val or s1.test != s2.test


# ---------------------------------------------------------------------------
# Coverage and leakage
# ---------------------------------------------------------------------------


def test_split_covers_all_episodes() -> None:
    split = _splitter().split(_EPISODE_IDS, seed=0)
    all_ids = set(split.train) | set(split.val) | set(split.test)
    assert all_ids == set(_EPISODE_IDS)


def test_split_has_no_leakage() -> None:
    split = _splitter().split(_EPISODE_IDS, seed=0)
    assert split.is_leak_free() is True


# ---------------------------------------------------------------------------
# Ratios and sizes
# ---------------------------------------------------------------------------


def test_split_default_ratios_approximate() -> None:
    split = _splitter().split(_EPISODE_IDS, seed=0, train_ratio=0.7, val_ratio=0.15)
    n = len(_EPISODE_IDS)
    assert len(split.train) >= round(n * 0.65)
    assert len(split.train) <= round(n * 0.75)


def test_split_records_seed_and_ratios() -> None:
    split = _splitter().split(_EPISODE_IDS, seed=7, train_ratio=0.6, val_ratio=0.2)
    assert split.seed == 7
    assert split.train_ratio == 0.6
    assert split.val_ratio == 0.2


# ---------------------------------------------------------------------------
# Edge cases and validation
# ---------------------------------------------------------------------------


def test_split_three_episode_minimum() -> None:
    """A list of 3 episodes yields non-empty train, val, and test partitions."""
    split = _splitter().split(["a", "b", "c"], seed=0)
    assert len(split.train) >= 1
    assert len(split.val) >= 1
    assert len(split.test) >= 1


def test_split_empty_list_raises() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        _splitter().split([], seed=0)


def test_split_bad_train_ratio_raises() -> None:
    with pytest.raises(ValueError, match="train_ratio"):
        _splitter().split(_EPISODE_IDS, seed=0, train_ratio=1.1)


def test_split_ratios_sum_over_one_raises() -> None:
    with pytest.raises(ValueError, match="train_ratio"):
        _splitter().split(_EPISODE_IDS, seed=0, train_ratio=0.8, val_ratio=0.3)
