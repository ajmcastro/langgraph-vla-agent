"""EpisodeSplitter — deterministic train/val/test partition of episode IDs."""

import random

from langgraph_vla_agent.datasets.episode import DatasetSplit


class EpisodeSplitter:
    """Produces a deterministic, leak-free train/val/test split.

    The split is computed by seeded shuffling of the episode ID list, then
    slicing at ratio-based index boundaries. Given the same episode list,
    ``seed``, ``train_ratio``, and ``val_ratio``, the output is always
    identical — this is a requirement for reproducible evaluation.

    Leakage check: the DatasetSplit.is_leak_free() invariant is guaranteed
    by construction (partition by disjoint index ranges of the shuffled list).
    """

    def split(
        self,
        episode_ids: list[str],
        *,
        seed: int,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> DatasetSplit:
        """Partition episode IDs into train / val / test.

        Parameters
        ----------
        episode_ids:
            All episode IDs to partition. Ordering does not matter; the list
            is copied and shuffled internally.
        seed:
            Random seed for the shuffle. Must be the same value to reproduce
            the same split.
        train_ratio:
            Fraction of episodes assigned to train (0 < train_ratio < 1).
        val_ratio:
            Fraction of episodes assigned to val (0 < val_ratio < 1).
            The test fraction is ``1 - train_ratio - val_ratio``.

        Returns
        -------
        DatasetSplit
            Partitioned episode IDs with seed and ratios recorded.

        Raises
        ------
        ValueError
            If ratios are out of range or the episode list is empty.
        """
        if not episode_ids:
            raise ValueError("episode_ids must not be empty")
        if not (0.0 < train_ratio < 1.0):
            raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
        if not (0.0 < val_ratio < 1.0):
            raise ValueError(f"val_ratio must be in (0, 1), got {val_ratio}")
        if train_ratio + val_ratio >= 1.0:
            raise ValueError(
                f"train_ratio + val_ratio must be < 1.0, got {train_ratio + val_ratio}"
            )

        shuffled = list(episode_ids)
        rng = random.Random(seed)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = max(1, round(n * train_ratio))
        n_val = max(1, round(n * val_ratio))
        # Ensure we don't exceed the list length; test gets the remainder
        n_train = min(n_train, n - 2)
        n_val = min(n_val, n - n_train - 1)

        train = shuffled[:n_train]
        val = shuffled[n_train : n_train + n_val]
        test = shuffled[n_train + n_val :]

        return DatasetSplit(
            train=train,
            val=val,
            test=test,
            seed=seed,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )
