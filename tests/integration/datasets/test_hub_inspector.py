"""Integration tests for HubDatasetInspector — requires network + [datasets] extra.

These tests are skipped automatically when huggingface_hub is not installed.
Run them explicitly with:
    uv sync --extra dev --extra datasets
    uv run pytest tests/integration -v
"""

import pytest

from langgraph_vla_agent.datasets.hub import HubDatasetInspector, hub_available

pytestmark = pytest.mark.integration

_HUB_ID = "lerobot/svla_so100_pickplace"

skip_if_no_hub = pytest.mark.skipif(
    not hub_available(),
    reason="huggingface_hub not installed — run: uv sync --extra dev --extra datasets",
)


@skip_if_no_hub
def test_hub_inspector_fetches_metadata() -> None:
    """Verify the Hub API returns metadata for the primary dataset."""
    inspector = HubDatasetInspector(_HUB_ID)
    info = inspector.fetch_info()

    assert info["hub_id"] == _HUB_ID
    assert info["sha"] is not None
    assert isinstance(info["sha"], str)
    assert len(str(info["sha"])) > 0


@skip_if_no_hub
def test_hub_inspector_builds_provenance() -> None:
    """Verify provenance record is constructed from Hub metadata."""
    inspector = HubDatasetInspector(_HUB_ID)
    prov = inspector.build_provenance(
        episodes=50,
        embodiment="so100",
        action_dim=6,
        obs_keys=["observation.image.front", "observation.state"],
        language_field="task",
        notes="Reference dataset from SmolVLA paper (arXiv:2506.01844).",
    )

    assert prov.hub_id == _HUB_ID
    assert prov.revision is not None
    assert prov.episodes == 50
    assert prov.action_dim == 6


def test_hub_inspector_raises_without_hub() -> None:
    """Without huggingface_hub, HubDatasetInspector raises ImportError."""
    if hub_available():
        pytest.skip("huggingface_hub is installed — can't test the ImportError path")

    with pytest.raises(ImportError, match="huggingface_hub"):
        HubDatasetInspector(_HUB_ID)
