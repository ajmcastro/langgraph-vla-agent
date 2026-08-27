#!/usr/bin/env python3
"""Dataset inspection script — Milestone 2.

Fetches metadata for the primary dataset from HuggingFace Hub without
downloading any data files (no parquet, no video frames).

Usage:
    uv run python scripts/inspect_dataset.py
    uv run python scripts/inspect_dataset.py --hub-id lerobot/svla_so100_pickplace

Requires the [datasets] extra:
    uv sync --extra dev --extra datasets
"""

import argparse
import json
import sys

_DEFAULT_HUB_ID = "lerobot/svla_so100_pickplace"


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a HuggingFace dataset (metadata only)")
    parser.add_argument(
        "--hub-id",
        default=_DEFAULT_HUB_ID,
        help=f"Dataset Hub ID (default: {_DEFAULT_HUB_ID})",
    )
    args = parser.parse_args()

    try:
        from langgraph_vla_agent.datasets.hub import HubDatasetInspector, hub_available
    except ImportError as e:
        print(f"[ERROR] Could not import langgraph_vla_agent: {e}", file=sys.stderr)
        sys.exit(1)

    if not hub_available():
        print(
            "[ERROR] huggingface_hub is not installed.\n"
            "Run:  uv sync --extra dev --extra datasets",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Inspecting: {args.hub_id}")
    print("(Fetching metadata only — no data files downloaded)\n")

    inspector = HubDatasetInspector(args.hub_id)

    info = inspector.fetch_info()
    print("=== Hub metadata ===")
    print(json.dumps(info, indent=2, default=str))

    print("\n=== Constructed provenance record ===")
    prov = inspector.build_provenance(
        episodes=50,
        embodiment="so100",
        action_dim=6,
        obs_keys=["observation.image.front", "observation.state"],
        language_field="task",
        notes="Reference dataset from SmolVLA paper (arXiv:2506.01844).",
    )
    print(prov.model_dump_json(indent=2))

    print(
        "\nTIP: Copy the 'revision' value to data/provenance/svla_so100_pickplace.yaml "
        "to lock the dataset version."
    )


if __name__ == "__main__":
    main()
