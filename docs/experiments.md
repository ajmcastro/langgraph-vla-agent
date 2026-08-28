# Experiment Log

A structured, reproducible record of every experiment run in this project.
Each entry states the conditions, results, what was proved, what was not proved,
and the exact command to reproduce the result.

Every metric is labelled with its evaluation mode. No result is extrapolated
beyond what the evaluation mode supports.

---

## Experiment 1 — SmolVLA Fine-tuning Comparison

**Milestone:** M4  
**Date:** 2026-08-27  
**Evaluation mode:** OFFLINE / REPLAY (fixture episodes)

### Conditions

| Condition | Model | Training |
|---|---|---|
| Base | `lerobot/smolvla_base` | None |
| Fine-tuned | `smolvla_so100_m4` | 10 000 steps, MPS, batch=8 |

**Dataset:** `lerobot/svla_so100_pickplace` (50 episodes, pick-and-place, SO-100 arm)  
**Training config:** [`configs/training/smolvla_so100.yaml`](../configs/training/smolvla_so100.yaml)  
**Provenance:** [`data/provenance/training/smolvla_so100_m4.yaml`](../data/provenance/training/smolvla_so100_m4.yaml)

### Results

| Metric | Base | Fine-tuned | Delta |
|---|---|---|---|
| L1 mean | 0.1893 | 0.1436 | −0.046 (−24%) |
| L2 mean | 0.2175 | 0.1812 | −0.036 (−17%) |
| L1 std | 0.0481 | 0.0225 | −0.026 (halved) |
| Training wall-clock | — | 2 h 23 min | — |
| Training cost | — | $0 (MPS) | — |

*Evaluated on 3 synthetic fixture episodes (not real held-out episodes from
`svla_so100_pickplace`).*

### What this proves

- The fine-tuning pipeline is correct end-to-end: config → `lerobot-train` → checkpoint → `compare_checkpoints`.
- `SmolVLAPolicyAdapter` can load and run inference from both the base and fine-tuned checkpoints.
- Fine-tuning on the dataset produces lower mean action prediction error on fixture episodes.
- L1 std halved, indicating more consistent predictions.

### What this does NOT prove

- That the −24% L1 reduction holds on real held-out episodes from `svla_so100_pickplace`.
  The fixture episodes are hand-crafted JSON files, not real recorded trajectories.
- That lower L1/L2 error translates to higher closed-loop task success (requires simulation
  or hardware — M7 provides toy closed-loop evidence, not physics-based evidence).
- That 10 000 steps on MPS is the optimal training recipe. A full-batch GPU run may differ.

### How to reproduce

```bash
make setup-vla                 # install [vla] extra
make validate-train-config     # dry-run (no GPU needed)
make train                     # run training on MPS (≈2h 23min)
make compare-checkpoints-vla   # base vs fine-tuned on fixture episodes
```

---

## Experiment 2 — Planning-Granularity Comparison (Mock Mode)

**Milestone:** M6  
**Date:** 2026-08-27  
**Evaluation mode:** MOCK (deterministic, scripted environment)

### Conditions

| Condition | Planner | Subtasks | Policy |
|---|---|---|---|
| `vla_only` | `VlaOnlyPlanner` | 1 (full goal) | `MockRobotPolicy` |
| `coarse_agentic` | `DeterministicPlanner("coarse")` | 2 | `MockRobotPolicy` |
| `fine_agentic` | `DeterministicPlanner("fine")` | 5 | `MockRobotPolicy` |

**Scenarios:** 5 success scenarios (pick-and-place goal variants)  
**Agent config:** `max_retries=2`, `max_replans=1`  
**Environment:** `MockEnvironment(SUCCEED_AT_STEP=2)` — terminates at step 2 regardless of action

### Results

| Condition | Episodes | Completed | Rate | Subtasks | Policy calls |
|---|---|---|---|---|---|
| `vla_only` | 5 | 5 | 100% | 1.0 | 1.0 |
| `coarse_agentic` | 5 | 5 | 100% | 2.0 | 2.0 |
| `fine_agentic` | 5 | 5 | 100% | 5.0 | 5.0 |

### What this proves

- All three conditions route through the identical compiled LangGraph graph.
- Subtask counts and policy-call counts are strictly ordered: vla_only < coarse < fine.
- The graph correctly handles goal decomposition, safety gate, retry/replan, and terminal states.
- Orchestration **cost** is measurable and differs systematically across planning granularity.

### What this does NOT prove

- That any condition improves real-world task success. `MockEnvironment` is scripted —
  it succeeds at step 2 regardless of policy output. All three complete at 100% because
  the environment is deterministic, not because any plan is better.
- That coarse or fine decomposition improves action quality (policy is MockRobotPolicy).

### How to reproduce

```bash
make setup-agent
make run-experiment          # success scenarios
make run-experiment-fail     # failure scenario (retry/replan paths)
```

---

## Experiment 3 — Planning-Granularity Comparison (Simulation Mode, Hard Scenario)

**Milestone:** M7  
**Date:** 2026-08-28  
**Evaluation mode:** SIMULATION (closed-loop toy scalar physics)

### Conditions

| Condition | Subtasks | Per-subtask threshold | Steps needed | Budget | Result |
|---|---|---|---|---|---|
| `vla_only` | 1 | 0.50 | ≈7 | 5 | FAILED |
| `coarse_agentic` | 2 | 0.25 | ≈4 | 5 | COMPLETED |
| `fine_agentic` | 5 | 0.10 | ≈2 | 5 | COMPLETED |

**Scenario:** `total_progress=0.5`, `progress_per_step=0.15`, `max_steps_per_subtask=5`, `noise_scale=0.0`, `seed=42`  
**Episodes:** 3 goals × 3 conditions = 9 total  
**Policy:** `MockRobotPolicy` (constant zero actions → `action_contribution=0.5`)  
**Agent config:** `max_retries=0`, `max_replans=0`

### Results

| Condition | Threshold | Completion rate | Notes |
|---|---|---|---|
| `vla_only` | 0.50 | 0% | 5 steps × 0.075 = 0.375 < 0.50 → FAILED |
| `coarse_agentic` | 0.25 | 100% | 4 steps × 0.075 = 0.300 ≥ 0.25 → COMPLETED |
| `fine_agentic` | 0.10 | 100% | 2 steps × 0.075 = 0.150 ≥ 0.10 → COMPLETED |

### What this proves

- `SimulationEnvironment` is closed-loop: actions affect world state, and the success
  predicate depends on accumulated progress — not on a scripted timer.
- Per-subtask budget constraints produce differentiated outcomes across planning conditions.
  This differentiation is **impossible in mock mode** (M6 shows all three at 100%).
- Finer decomposition distributes the same total task difficulty into smaller, individually
  achievable per-subtask targets.

### What this does NOT prove

- Real physics. `SimulationEnvironment` is a single scalar `progress ∈ [0, 1]`.
  No joint positions, no rigid-body dynamics, no camera rendering.
- That `MockRobotPolicy` is representative of SmolVLA or any real policy.
  The differentiation is entirely due to threshold scaling, not policy quality.
- Sim-to-real transfer. The toy model does not model any physical phenomenon present
  in the SO-100 hardware task.
- Statistical significance. N=3 episodes per condition is below the N ≥ 10 threshold
  for hypothesis tests.

### How to reproduce

```bash
make setup-agent
make run-simulation           # easy mode (all conditions succeed)
make run-simulation-hard      # hard mode (vla_only fails)
```

---

## Reproduce all experiments with the portfolio demo

```bash
make setup-agent
make run-demo                 # modes 1 (replay), 2 (mock agent), 3 (simulation)
```
