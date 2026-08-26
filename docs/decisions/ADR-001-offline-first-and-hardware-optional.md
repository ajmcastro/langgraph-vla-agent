# ADR-001: Hardware-optional, offline-first architecture

**Status:** Accepted  
**Date:** 2026-08-26  
**Deciders:** Antonio Castro

---

## Context

This project investigates whether agentic orchestration improves multi-step manipulation task success for a VLA policy. The natural validation path is closed-loop execution on a physical robot. However:

1. No physical robot (SO-101 or equivalent) is available during the primary development period.
2. Purchasing, assembling, calibrating, and safely operating a robot arm introduces weeks of non-research work and physical-safety obligations that are not the focus of this project.
3. The research question — whether agentic planning helps — can be explored meaningfully through offline/replay evaluation and (optionally) simulation before hardware is needed.
4. A portfolio project must remain runnable and demonstrable to reviewers who also lack hardware.

---

## Decision

**Mock and offline/replay evaluation are first-class backends, not fallbacks.**

Concretely:
- `MockRobotPolicy` and `MockEnvironment` are required implementations. They must be capable of exercising all graph nodes, recovery paths, and safety gates without any external dependencies.
- `ReplayRobotPolicy` and `ReplayEnvironment` are required for offline evaluation against public datasets.
- The `SmolVLAPolicyAdapter` and all hardware adapters are optional dependencies gated behind extras (`[vla]`, future `[hardware]`).
- No milestone acceptance criterion requires a GPU, simulator, cloud account, or physical robot, except those explicitly labelled as GPU/cloud/hardware milestones (4, 7, optional hardware milestone).
- Hardware adapters live in an isolated module (`policies/hardware/`, `environments/hardware/`) and are never imported by core agent, evaluation, or test code.
- Hardware-in-the-loop tests are in `tests/hardware/` and excluded from default CI with a pytest marker.

---

## Consequences

**Positive:**
- The project can be developed, tested, and demonstrated entirely on a laptop.
- All graph logic, planner behavior, safety gates, and orchestration decisions are unit-testable with deterministic mocks.
- Adding hardware later requires only a new adapter implementation — no changes to the agent, planner, or evaluation code.
- Evaluation claims are honest: mock results prove software correctness, replay results prove prediction quality, simulation results prove closed-loop behavior under specific conditions.

**Negative / trade-offs:**
- Offline and mock evaluation cannot fully validate closed-loop robot performance. This limitation must be disclosed in every result section.
- The `ReplayRobotPolicy` cannot model the counterfactual consequences of deviating from the recorded trajectory. Results are indicative, not conclusive.
- Without closed-loop feedback, some failure modes (contact dynamics, sensor noise, actuator lag) are invisible until hardware experiments.

**Mitigations:**
- All evaluation results carry an explicit mode label.
- The research question is scoped to what can be answered honestly in each mode (planning logic in mock; prediction quality in replay; closed-loop behavior in sim or hardware).
- `docs/evaluation.md` documents precisely what each mode can and cannot prove.

---

## Rejected alternatives

**Require a physical robot from the start:** Would block the entire project on hardware procurement, assembly, and safety review. The agent and orchestration logic does not require hardware to be designed or verified.

**Use simulation as the first-class backend instead of replay:** Simulation (MuJoCo via gym-pusht / gym-aloha) adds a large dependency, requires task definitions that may differ from the SmolVLA training distribution, and is harder to set up on Apple Silicon. Simulation is preserved as an optional Milestone 7 if the offline evidence warrants it.

**Use replay evaluation only (no mock layer):** Replay requires dataset downloads and schema compatibility. Mock is strictly lighter and sufficient for all behavioral tests. Both are necessary; neither replaces the other.
