# LangGraph VLA Agent

An educational, production-quality open-source project demonstrating how an **agentic planning layer** (LangGraph + LLM) can orchestrate a **Vision-Language-Action (VLA) sensorimotor policy** (SmolVLA via Hugging Face LeRobot) for multi-step robot manipulation tasks.

**Key components:**
- [**LeRobot**](https://github.com/huggingface/lerobot) — HuggingFace's open-source robotics library providing datasets, training pipelines, and pre-trained robot policies.
- [**SmolVLA**](https://huggingface.co/lerobot/smolvla_base) ([paper](https://arxiv.org/abs/2506.01844)) — A small Vision-Language-Action model that takes camera images and a natural-language instruction and outputs joint-space robot actions. Designed for SO-100 robot arms.
- [**LangGraph**](https://github.com/langchain-ai/langgraph) — A framework for building stateful, multi-step LLM agent graphs. Used here to decompose goals into subtasks and orchestrate policy execution.

> **No physical robot required.** The primary execution path uses public LeRobot-compatible datasets, offline/replay evaluation, and deterministic mocks. Hardware integration is optional future work behind isolated adapters.

---

## Research question

> When does adding an agentic planning and orchestration layer on top of a learned VLA policy improve multi-step task completion compared with giving the entire natural-language task directly to the VLA?

The project evaluates potential gains in success rate, failure recovery, language generalisation, and interpretability — and measures costs in latency, LLM calls, and complexity — using three planning levels:

| Level | Description |
|---|---|
| **VLA-only** | Full multi-step instruction passed directly to the policy |
| **Coarse agentic** | LangGraph decomposes into a small set of meaningful manipulation subtasks |
| **Fine agentic** | LangGraph decomposes into smaller physical skills (no actuator-level commands) |

All evaluation is performed in mock, offline/replay, or simulation mode. Claims are labelled by their evaluation mode and are never extrapolated to physical-robot performance without explicit hardware experiments.

---

## Architecture

```
Natural-language goal
        ↓
LangGraph (goal/subtask timescale)
  understand_goal → create_plan → select_subtask → safety_check
        ↓
Executor (observation/action timescale)
  obs → RobotPolicy.act() → action → RobotEnvironment.step()
        ↓
RobotPolicy abstraction
  MockRobotPolicy | ReplayRobotPolicy | SmolVLAPolicyAdapter
        ↓
RobotEnvironment abstraction
  MockEnvironment | ReplayEnvironment | [SimulationEnvironment]
        ↓
Observation + result → verify → retry / replan / complete
```

**The LLM layer never generates joint torques, servo positions, or high-frequency trajectories.** It operates exclusively at the goal and subtask level.

---

## Results to date

M4 (fine-tuning infrastructure + training run) is the furthest milestone with measurable results. All numbers are from offline/replay evaluation on synthetic fixture episodes — see [docs/evaluation.md](docs/evaluation.md) for what this means and what it does not prove.

| Milestone | What ran | Key result |
|---|---|---|
| M1 — Mock loop | Deterministic mock executor | Graph routing, retry, safety gate all testable without GPU |
| M2 — Replay backend | Fixture episode replayer | Episode splitter, replay policy, and environment infrastructure verified |
| M3 — SmolVLA baseline | SmolVLA adapter on fixture episodes | Base model produces valid 6-D float32 actions; accuracy not yet meaningful (fixture data only) |
| M4 — Fine-tuning | 10 k-step MPS training run on `svla_so100_pickplace` | L1 error: 0.1893 → 0.1436 (−24%) vs base; L1 std halved |

**M4 caveat:** The −24% L1 improvement is measured on 3 synthetic fixture episodes, not on real held-out episodes from `svla_so100_pickplace`. Lower prediction error on fixtures does not prove closed-loop task success. See [`data/provenance/training/smolvla_so100_m4.yaml`](data/provenance/training/smolvla_so100_m4.yaml) for full provenance.

The core research comparison (VLA-only vs coarse agentic vs fine agentic) is Milestone 6 and has not yet run.

---

## Environments

| Environment | Purpose | Status |
|---|---|---|
| MacBook Pro M4 Max (64 GB) | Development, tests, mock/replay eval, dataset inspection, MPS training | Primary |
| Cloud NVIDIA GPU | Large-scale training, full-batch fine-tuning, checkpoint export | Optional (M4+) |
| Physical robot (SO-101) | Hardware validation | Future work only |

---

## Quick start

### Prerequisites

- [uv](https://docs.astral.sh/uv/) ≥ 0.6 (`brew install uv` on macOS)
- Python 3.12 (installed automatically by uv)

### Setup

```bash
git clone https://github.com/<your-org>/langgraph-vla-agent.git
cd langgraph-vla-agent

# Install core + dev deps (no GPU, no model download required)
make setup

# Verify everything works
make check
```

### Running tests

```bash
make test-unit        # fast, no external deps
make test             # all non-hardware tests
```

### Fine-tuning SmolVLA (Milestone 4)

```bash
# Validate the training config without running training (no GPU needed):
make validate-train-config

# Compare two checkpoints in fixture mode (no GPU, no trained model needed):
make compare-checkpoints

# Run training locally (requires [vla] extra + CUDA or MPS GPU):
make train

# Compare real checkpoints after training (uses the final step-10000 checkpoint):
make compare-checkpoints-vla
```

For cloud GPU training via HF Jobs, set `hf_jobs_target` in [`configs/training/smolvla_so100.yaml`](configs/training/smolvla_so100.yaml) to your chosen hardware flavor (e.g. `nvidia-l40s-x1`), then run `make train-cloud`.

**Completed M4 run (2026-08-27, Apple M4 Max MPS):**

| Metric | Base | Fine-tuned | Delta |
|---|---|---|---|
| L1 mean | 0.1893 | 0.1436 | -0.046 (−24%) |
| L2 mean | 0.2175 | 0.1812 | -0.036 |
| Wall-clock | — | — | 2 h 23 min / 10 k steps |

*Evaluated on 3 synthetic fixture episodes. See [`data/provenance/training/smolvla_so100_m4.yaml`](data/provenance/training/smolvla_so100_m4.yaml) for full provenance.*

### Running offline policy evaluation (Milestone 3)

```bash
# Mock evaluation — no GPU or dataset needed
make evaluate-mock

# Replay evaluation on fixture episodes — no GPU needed
make evaluate-replay

# VLA evaluation — requires [vla] extra and model download (~500 MB)
make setup-vla
make evaluate-policy
```

### Adding optional extras (later milestones)

```bash
make setup-datasets   # adds huggingface_hub for dataset inspection (Milestone 2+)
make setup-agent      # adds LangGraph + langchain-core (Milestone 5+)
make setup-vla        # adds LeRobot + SmolVLA — large download (Milestone 3+)
```

For the VLA extra on macOS, also install ffmpeg:
```bash
brew install ffmpeg
```

For the VLA extra on Linux with CUDA:
```bash
uv pip install --torch-backend cu128 lerobot
```

---

## Milestones

| # | Title | Status |
|---|---|---|
| 0 | Foundation and verified project plan | ✅ Complete |
| 1 | Domain contracts and deterministic mock loop | ✅ Complete |
| 2 | Public dataset inspection and replay backend | ✅ Complete |
| 3 | SmolVLA baseline | ✅ Complete |
| 4 | Cloud GPU fine-tuning | ✅ Complete (trained on MPS, 24% L1 improvement) |
| 5 | LangGraph orchestration | Pending |
| 6 | Planning-granularity experiments | Pending |
| 7 | Optional closed-loop simulation | Pending |
| 8 | Portfolio hardening | Pending |

See [`docs/PROJECT_SPEC.md`](docs/PROJECT_SPEC.md) for the full specification and [`docs/RATIONALE_PER_MILESTONE.md`](docs/RATIONALE_PER_MILESTONE.md) for design rationale.

---

## Documentation

| File | Contents |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | System architecture and layer boundaries |
| [`docs/evaluation.md`](docs/evaluation.md) | Evaluation modes, metrics, and honesty constraints |
| [`docs/data.md`](docs/data.md) | Dataset strategy, candidates, provenance |
| [`docs/safety.md`](docs/safety.md) | Safety design (software guards and future hardware concerns) |
| [`docs/decisions/`](docs/decisions/) | Architecture Decision Records (ADRs) |

---

## License

MIT — see [LICENSE](LICENSE).
