# Data Strategy

## Principles

1. **No locally collected data required.** The primary path uses public LeRobot-compatible datasets.
2. **Never commit large files.** Datasets, checkpoints, and generated artifacts are `.gitignore`d. Only metadata, small fixtures (≤ a few KB), and provenance records are committed.
3. **Every dataset reference must record provenance:** HuggingFace Hub ID, revision/commit hash, license, episode count, action/observation schema, embodiment, and known limitations.
4. **Splits must be deterministic and leak-free.** Evaluation episodes and any paraphrases of evaluation language must not appear in training data.

---

## Dataset candidates (Milestone 0 — not yet downloaded)

The following candidates are identified from public documentation and verified license information. Full schema inspection, compatibility checks, and final selection happen in Milestone 2.

### Primary candidate: `lerobot/svla_so100_pickplace`

| Property | Value |
|---|---|
| Hub ID | `lerobot/svla_so100_pickplace` |
| Robot | SO100 (SO-100 follower arm) |
| Task | Pick-and-place (cube → bin) |
| Episodes | 50 (10 per cube position × 5 positions) |
| Language | Natural language task descriptions per episode |
| Cameras | Front camera (640×480, 30 fps) — dataset key `observation.image.front` |
| Action space | SO100 joint positions (6-DOF arm) |
| Provenance | Used in the SmolVLA paper (arXiv:2506.01844) |
| License | To be verified in M2 — expected Apache 2.0 or MIT per HF LeRobot norms |
| Compatibility | Designed for SmolVLA fine-tuning; action/obs modalities match `smolvla_base` |
| Risks | 50 episodes is small for generalisation experiments; scene diversity limited to 5 cube positions |

> **M3 finding — image key mismatch:** `smolvla_base` (lerobot 0.6.1) expects
> `observation.images.camera1/2/3` at `(3, 256, 256)`, not `observation.image.front` at
> `(3, 480, 640)`. `SmolVLAPolicyAdapter` introspects `model.config.image_features` at
> runtime so it always sends whatever the loaded checkpoint declares. When running against
> fixture episodes (which have no camera images) the adapter creates dummy black images of
> the correct shape. This discrepancy between dataset and base-model camera conventions
> must be resolved before M4 fine-tuning — likely by using the correct dataset split that
> the base model was actually trained on, or by preprocessing the dataset images to the
> model's expected resolution and key names.

**Why this dataset?** It is the reference dataset from the SmolVLA paper, explicitly designed for the `smolvla_base` checkpoint we will use in Milestone 3. Using it gives a direct apples-to-apples comparison between the base model and our fine-tuned/orchestrated variants.

### Backup candidate: `lerobot/pusht`

| Property | Value |
|---|---|
| Hub ID | `lerobot/pusht` |
| Task | Push-T (2D planar pushing) |
| Episodes | ~200 |
| Language annotations | **No** — single task, no per-episode language |
| License | MIT |
| Compatibility | Requires `gym-pusht` sim; action space does not match SmolVLA (2D vs 6-DOF arm) |
| Risks | No language annotations makes it unsuitable for VLA evaluation; different embodiment from SmolVLA training data |

**Status:** Suitable for simulation environment testing (Milestone 7) but not for SmolVLA-based VLA evaluation. Listed as a backup for sim-only experiments.

---

## Open questions (to resolve in Milestone 2)

- [ ] Exact license of `lerobot/svla_so100_pickplace` — confirm Apache 2.0 or MIT
- [ ] Exact Hub revision/commit hash — needed for provenance lock
- [ ] Does the dataset include evaluation/held-out episodes, or must we create a split?
- [ ] Are language annotations per-episode or per-task (single string)?
- [ ] Dataset size on disk (compressed parquet + video frames)
- [ ] Compatibility with `lerobot[dataset]` extra schema for streaming and batching

---

## Data workflow (planned)

```
1. Metadata-only inspection (make inspect-data)
   └─ Fetch Hub metadata, episode count, schema, license, size estimate
   └─ No data downloaded

2. Tiny sample download (M2)
   └─ First N episodes to a local cache in data/cache/ (gitignored)
   └─ Validate schema: obs keys, action shape, timestamps, language field

3. Split construction (M2)
   └─ Deterministic train/val/test split with fixed seed
   └─ Leakage check: no eval episode appears in train

4. Replay scenario construction (M2)
   └─ Held-out episodes become replay scenarios for ReplayRobotPolicy
   └─ Scenario metadata committed to data/fixtures/

5. Fine-tuning data pipeline (M4, cloud GPU)
   └─ Full dataset streamed from Hub during training job
   └─ Preprocessing config committed to configs/training/
```

---

## Provenance record format (template)

```yaml
# data/provenance/<dataset-name>.yaml  (committed; no large data files)
dataset:
  hub_id: lerobot/svla_so100_pickplace
  revision: <commit-hash>          # filled in M2
  license: <SPDX-identifier>       # filled in M2
  episodes: 50
  embodiment: so100
  action_dim: 6
  obs_keys: [observation.image.front, observation.state]
  language_field: task
  download_date: <ISO-8601>
  checksum: <sha256 of metadata file>  # not of dataset itself
  notes: Reference dataset from SmolVLA paper (arXiv:2506.01844)
```

---

## What we will NOT do

- Collect our own demonstrations (no hardware available)
- Commit dataset files (even small parquet files)
- Assume SO-100 dataset compatibility with non-SO-100 SmolVLA checkpoints without verification
- Report training or evaluation results without identifying the exact dataset revision used
