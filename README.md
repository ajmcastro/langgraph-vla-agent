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

M7 is the current completed milestone. M4 (fine-tuning) is the furthest milestone with measurable numbers — see [docs/evaluation.md](docs/evaluation.md) for what each result means and what it does not prove.

| Milestone | What ran | Key result |
|---|---|---|
| M1 — Mock loop | Deterministic mock executor | Graph routing, retry, safety gate all testable without GPU |
| M2 — Replay backend | Fixture episode replayer | Episode splitter, replay policy, and environment infrastructure verified |
| M3 — SmolVLA baseline | SmolVLA adapter on fixture episodes | Base model produces valid 6-D float32 actions; accuracy not yet meaningful (fixture data only) |
| M4 — Fine-tuning | 10 k-step MPS training run on `svla_so100_pickplace` | L1 error: 0.1893 → 0.1436 (−24%) vs base; L1 std halved |
| M5 — LangGraph orchestration | Full StateGraph with DeterministicPlanner + MockRobotPolicy | Graph routing, retry/replan/fail paths, and safety gate verified in mock mode; 249 tests pass |
| M6 — Granularity experiments | VlaOnlyPlanner + 3-condition experiment infrastructure | Orchestration cost confirmed: vla_only=1 policy call, coarse=2, fine=5; all conditions 100% on mock success scenarios; 270 tests pass |
| M7 — Closed-loop simulation | SimulationEnvironment + 3-condition simulation experiment | Hard scenario: vla_only FAILS (threshold unreachable in budget), coarse/fine SUCCEED; first result mock mode cannot produce; 293 tests pass |

**M4 caveat:** The −24% L1 improvement is measured on 3 synthetic fixture episodes, not on real held-out episodes from `svla_so100_pickplace`. Lower prediction error on fixtures does not prove closed-loop task success. See [`data/provenance/training/smolvla_so100_m4.yaml`](data/provenance/training/smolvla_so100_m4.yaml) for full provenance.

**M5 note:** All results are in mock evaluation mode. The graph is wired to `MockRobotPolicy` and `MockEnvironment` — no LLM key or GPU needed.

**M6 note:** The 3-condition experiment (vla_only / coarse_agentic / fine_agentic) runs end-to-end through the compiled LangGraph graph. In mock mode all three complete at 100% on success scenarios — the meaningful comparison is orchestration *cost* (policy calls, subtask count), not success rate. Real performance differences require simulation or hardware — M7 provides toy closed-loop evidence (scalar progress model, no physics), but connecting to a real physics simulator or real policy checkpoint is future work.

**M7 note:** `SimulationEnvironment` is a toy scalar progress model — no external simulator, GPU, or dataset. Actions affect world state: full-positive actions produce full-speed progress, zero actions produce half-speed. In the hard scenario (total_progress=0.5, max_steps=5), vla_only needs 7 steps to reach its threshold but only has a budget of 5 → FAILED. Coarse (threshold=0.25, needs 4 steps) and fine (threshold=0.10, needs 2 steps) succeed. This differentiation cannot be produced in mock mode. `SimulationEnvironment` is not a physics simulator; see [docs/evaluation.md](docs/evaluation.md) for what M7 does and does not prove.

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

### Running the simulation experiment (Milestone 7)

```bash
# Install the [agent] extra (adds LangGraph + langchain-core):
make setup-agent

# Easy mode — all three conditions succeed (mirrors mock mode):
make run-simulation

# Hard mode — vla_only FAILS, coarse/fine SUCCEED (closed-loop differentiation):
make run-simulation-hard

# Custom parameters:
uv run python scripts/run_simulation.py --progress 0.4 --steps 6
uv run python scripts/run_simulation.py --hard --noise 0.05 --quiet
```

No external simulator, GPU, or dataset needed. The output table shows per-condition completion rates and per-subtask thresholds, plus an explanation of what the result proves and what it does not.

### Running the planning-granularity experiment (Milestone 6)

```bash
# Install the [agent] extra (adds LangGraph + langchain-core):
make setup-agent

# Run the 3-condition experiment (5 success scenarios, ~15 episodes total):
make run-experiment

# Run with a failure scenario to see retry/replan paths in action:
make run-experiment-fail

# Custom options:
uv run python scripts/run_experiment.py --max-retries 3 --max-replans 2
uv run python scripts/run_experiment.py --fail-scenario --quiet
```

No LLM key or GPU needed. The output table shows orchestration cost (policy calls, subtask counts) across the three conditions, plus a disclaimer explaining what the mock results do and do not prove.

### Running the LangGraph agent (Milestone 5)

```bash
# Install the [agent] extra (adds LangGraph + langchain-core):
make setup-agent

# Run a single pick-and-place episode in coarse mode (2 subtasks):
make evaluate-agent

# Run in fine-grained mode (5 subtasks):
make evaluate-agent-fine

# Run with a custom goal:
uv run python scripts/run_agent.py --goal "pick up the cube and place it in the bin"
```

No LLM key or GPU is needed — the agent uses `DeterministicPlanner` and `MockRobotPolicy` by default. The output includes what the run proves and what it does not prove.

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
| 5 | LangGraph orchestration | ✅ Complete (full StateGraph with retry/replan/safety; 249 tests) |
| 6 | Planning-granularity experiments | ✅ Complete (VlaOnlyPlanner + 3-condition experiment; 270 tests) |
| 7 | Optional closed-loop simulation | ✅ Complete (SimulationEnvironment + hard-scenario differentiation; 293 tests) |
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
