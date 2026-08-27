"""Dataset adapters and replay infrastructure for langgraph-vla-agent.

Public surface for M2:
    ReplayStep, ReplayEpisode   — recorded trajectory domain models
    DatasetProvenance           — provenance record for reproducibility
    DatasetSplit                — train/val/test partition of episode IDs
    EpisodeStore                — structural Protocol for episode sources
    FixtureEpisodeStore         — reads committed JSON fixtures (no network)
    EpisodeSplitter             — deterministic seed-based split
    hub_available               — checks if huggingface_hub is installed
    HubDatasetInspector         — optional, requires [datasets] extra
"""

from langgraph_vla_agent.datasets.episode import (
    DatasetProvenance,
    DatasetSplit,
    ReplayEpisode,
    ReplayStep,
)
from langgraph_vla_agent.datasets.hub import HubDatasetInspector, hub_available
from langgraph_vla_agent.datasets.splits import EpisodeSplitter
from langgraph_vla_agent.datasets.store import EpisodeStore, FixtureEpisodeStore

__all__ = [
    "DatasetProvenance",
    "DatasetSplit",
    "EpisodeSplitter",
    "EpisodeStore",
    "FixtureEpisodeStore",
    "HubDatasetInspector",
    "ReplayEpisode",
    "ReplayStep",
    "hub_available",
]
