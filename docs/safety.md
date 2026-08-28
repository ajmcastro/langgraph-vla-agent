# Safety Design

## Scope and honesty

This project operates primarily in mock, offline/replay, and optional simulation modes. The software safety guards described here are testable in those modes. They are **not** a substitute for physical safety engineering, which requires hardware validation, calibration, joint-limit firmware, collision detection, and supervised first execution.

Physical safety concerns are documented in the "Future hardware considerations" section below. Do not implement fake hardware safety that gives false confidence.

---

## Software safety layer (implemented in M5 — mock mode)

### Instruction allowlist and blocklist (implemented: `SafetyChecker`)

`src/langgraph_vla_agent/agent/safety.py` implements a two-rule gate applied to every subtask instruction before execution:

1. **Blocked terms:** if the instruction contains any word from a blocklist (e.g. `human`, `person`, `face`, `sharp`, `blade`, `knife`, `fire`, `weapon`), the subtask is immediately rejected.
2. **Allowed verbs:** the instruction must contain at least one verb from an allowlist (e.g. `approach`, `grasp`, `pick`, `lift`, `place`, `push`, `pull`, `open`, `close`). Instructions with no recognised manipulation verb are rejected.

Rejection sets `final_status = AgentStatus.SAFETY_STOP` and halts the episode. The safety gate is wired between `select_next_subtask` and `execute_policy` in the graph; it can be disabled per episode via `AgentConfig(safety_check_enabled=False)`.

### Bounded execution (implemented: `AgentConfig`)

Each episode enforces hard limits to prevent runaway execution:

| Limit | Default | Config field |
|---|---|---|
| Max retries per subtask | 2 | `AgentConfig.max_retries` |
| Max replan cycles per episode | 1 | `AgentConfig.max_replans` |
| Max action steps per subtask | 200 | `ExecutorConfig.max_steps` |

Exhausting retries and replans sets `final_status = AgentStatus.FAILED`. Exceeding `max_steps` per subtask returns `ExecutionStatus.MAX_STEPS_EXCEEDED`, which is treated as a subtask failure and flows into the retry/replan logic.

### Action schema and range validation (implemented: `Executor`)

All actions produced by `RobotPolicy.act()` are validated against a Pydantic schema (finite float32 values) before being passed to `RobotEnvironment.step()`. Validation failures set `ExecutionStatus.INVALID_ACTION`. The executor does not retry invalid actions automatically — a retry requires the graph's `diagnose_failure` node to authorise it.

### Fail-closed behavior (implemented)

When any of the following occur, the agent sets a terminal `AgentStatus` and stops:

- Goal text is empty → `FAILED`
- Planner raises `PlanningError` and replans are exhausted → `FAILED`
- Safety gate rejects the subtask instruction → `SAFETY_STOP`
- Both retry budget and replan budget are exhausted → `FAILED`

### Cancellation

The agent supports explicit cancellation via a `CANCEL` signal injected into graph state. Cancellation transitions immediately to a `CANCELLED` terminal state without waiting for the current step to complete (in mock/sim) or after the current action completes safely (hardware, future).

---

## Observability and audit

Safety decisions are logged as structured events with:
- `run_id`, `episode_id`, `subtask_id`
- `safety_event_type` (ALLOWED, BLOCKED, LIMIT_EXCEEDED, etc.)
- `trigger_reason` and `triggering_value`
- Timestamp and evaluation mode

These events are separate from the general execution log and are never redacted. They form the audit trail for safety review.

---

## Future hardware considerations

**None of the following are implemented. They are documented here for reference when hardware becomes available.**

- **Joint limits:** firmware-enforced position, velocity, and torque limits per motor — must be set before any powered motion
- **Collision detection:** workspace bounding box, obstacle models, and self-collision checks
- **Emergency stop:** hardware E-stop accessible by the human operator at all times; checked before each action command
- **Calibration validation:** verify calibration before each session; halt if calibration is stale or out of tolerance
- **Supervised first execution:** all new tasks must be executed slowly under direct human supervision in the first session
- **Workspace constraint:** physical workspace must be cleared and camera field of view confirmed before each episode

Physical safety validation is not a software milestone. It requires a safety review with the hardware in hand.
