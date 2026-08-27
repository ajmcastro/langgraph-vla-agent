"""EpisodeStore — Protocol and FixtureEpisodeStore implementation."""

from pathlib import Path
from typing import Protocol, runtime_checkable

from langgraph_vla_agent.datasets.episode import ReplayEpisode


@runtime_checkable
class EpisodeStore(Protocol):
    """Structural interface for episode sources.

    Implementations
    ---------------
    FixtureEpisodeStore  — reads committed JSON fixtures (no network, M2)
    HubEpisodeStore      — streams from HuggingFace Hub (M3, requires [vla])
    """

    def list_episodes(self) -> list[str]:
        """Return a sorted list of available episode IDs."""
        ...

    def load_episode(self, episode_id: str) -> ReplayEpisode:
        """Load and return the episode with the given ID.

        Raises
        ------
        FileNotFoundError
            If no episode with that ID exists in this store.
        """
        ...


class FixtureEpisodeStore:
    """Loads replay episodes from committed JSON fixture files.

    Each file in ``fixture_dir`` must be named ``<episode_id>.json`` and
    contain a JSON-serialised ReplayEpisode. No network or optional deps
    are required.

    Parameters
    ----------
    fixture_dir:
        Path to the directory containing ``*.json`` episode fixture files.
        Typically ``data/fixtures/episodes/`` relative to the project root.
    """

    def __init__(self, fixture_dir: Path) -> None:
        self._fixture_dir = fixture_dir

    def list_episodes(self) -> list[str]:
        """Return a sorted list of episode IDs (file stems)."""
        return sorted(p.stem for p in self._fixture_dir.glob("*.json"))

    def load_episode(self, episode_id: str) -> ReplayEpisode:
        """Load the episode from its JSON file.

        Parameters
        ----------
        episode_id:
            Must match the file stem (filename without ``.json``).

        Raises
        ------
        FileNotFoundError
            If ``<fixture_dir>/<episode_id>.json`` does not exist.
        """
        path = self._fixture_dir / f"{episode_id}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Episode {episode_id!r} not found in fixture directory {self._fixture_dir}"
            )
        return ReplayEpisode.model_validate_json(path.read_text())
