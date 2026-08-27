"""Optional HubDatasetInspector — requires the [datasets] extra.

Install:
    uv sync --extra dev --extra datasets

This module is deliberately not imported by any core module. It is used
only by scripts/inspect_dataset.py and integration tests that are skipped
when huggingface_hub is not installed.
"""

from __future__ import annotations

try:
    from huggingface_hub import DatasetCard, HfApi

    _hub_available = True
except ImportError:
    _hub_available = False

from langgraph_vla_agent.datasets.episode import DatasetProvenance


def hub_available() -> bool:
    """Return True if huggingface_hub is installed."""
    return _hub_available


class HubDatasetInspector:
    """Fetches dataset metadata from HuggingFace Hub without downloading data.

    Only the Hub API metadata endpoint is called — no parquet files or video
    frames are downloaded. Requires a network connection and the [datasets]
    extra (huggingface_hub).

    Parameters
    ----------
    hub_id:
        Dataset repository ID, e.g. ``"lerobot/svla_so100_pickplace"``.
    """

    def __init__(self, hub_id: str) -> None:
        if not _hub_available:
            raise ImportError(
                "huggingface_hub is required for HubDatasetInspector. "
                "Install the [datasets] extra:  uv sync --extra dev --extra datasets"
            )
        self._hub_id = hub_id
        self._api: HfApi = HfApi()

    def fetch_info(self) -> dict[str, object]:
        """Return raw metadata from the Hub API (no data downloaded).

        Returns a dict with at minimum:
          - ``hub_id``: the dataset ID
          - ``sha``: latest commit hash (the revision to record in provenance)
          - ``card_data``: parsed YAML front matter from the dataset README
        """
        info = self._api.dataset_info(self._hub_id)
        return {
            "hub_id": self._hub_id,
            "sha": info.sha,
            "private": info.private,
            "card_data": info.card_data.to_dict() if info.card_data else {},  # type: ignore[no-untyped-call]
        }

    def fetch_card_text(self) -> str:
        """Return the raw dataset card (README.md) text."""
        card: DatasetCard = DatasetCard.load(self._hub_id)
        return str(card.content)

    def build_provenance(
        self,
        *,
        episodes: int,
        embodiment: str,
        action_dim: int,
        obs_keys: list[str],
        language_field: str,
        notes: str = "",
    ) -> DatasetProvenance:
        """Construct a DatasetProvenance record from Hub metadata + caller-supplied schema.

        The schema fields (episodes, embodiment, action_dim, obs_keys,
        language_field) must be supplied by the caller because the Hub API
        does not expose them in a structured way for all datasets.
        """
        info = self.fetch_info()
        card_data = info.get("card_data", {})
        license_val = None
        if isinstance(card_data, dict):
            raw_license = card_data.get("license")
            if isinstance(raw_license, list) and raw_license:
                license_val = str(raw_license[0])
            elif isinstance(raw_license, str):
                license_val = raw_license

        return DatasetProvenance(
            hub_id=self._hub_id,
            revision=str(info.get("sha")) if info.get("sha") else None,
            license_spdx=license_val,
            episodes=episodes,
            embodiment=embodiment,
            action_dim=action_dim,
            obs_keys=obs_keys,
            language_field=language_field,
            notes=notes,
        )
