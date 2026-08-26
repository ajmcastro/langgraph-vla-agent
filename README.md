# LangGraph VLA Agent

An educational, production-quality open-source project demonstrating how an **agentic planning layer** (LangGraph + LLM) can orchestrate a **Vision-Language-Action (VLA) sensorimotor policy** (SmolVLA via Hugging Face LeRobot) for multi-step robot manipulation tasks.

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

## Environments

| Environment | Purpose | Status |
|---|---|---|
| MacBook Pro M4 Max (64 GB) | Development, tests, mock/replay eval, dataset inspection | Primary |
| Cloud NVIDIA GPU | SmolVLA fine-tuning, batch eval, checkpoint export | Milestone 4+ |
| Physical robot (SO-101) | Optional hardware validation | Future work only |

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

### Adding optional extras (later milestones)

```bash
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
| 2 | Public dataset inspection and replay backend | Pending |
| 3 | SmolVLA baseline | Pending |
| 4 | Cloud GPU fine-tuning | Pending |
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
