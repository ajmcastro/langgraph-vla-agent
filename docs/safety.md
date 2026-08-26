# Safety Design

## Scope and honesty

This project operates primarily in mock, offline/replay, and optional simulation modes. The software safety guards described here are testable in those modes. They are **not** a substitute for physical safety engineering, which requires hardware validation, calibration, joint-limit firmware, collision detection, and supervised first execution.

Physical safety concerns are documented in the "Future hardware considerations" section below. Do not implement fake hardware safety that gives false confidence.

---

## Software safety layer (implemented in mock and replay modes)

### Task allowlist

The agent accepts only tasks from an allowlisted set of skill categories. Tasks outside the allowlist trigger a `TASK_NOT_ALLOWED` safety stop before any planning or execution begins. This prevents the orchestration layer from being used as a general-purpose actuator controller.

Example allowlisted categories (configurable per deployment):
- pick-and-place
- push
- grasp-and-release
- open/close gripper

### Bounded execution

Each episode enforces hard limits to prevent runaway execution:

| Limit | Default | Config key |
|---|---|---|
| Max retries per subtask | 3 | `safety.max_retries` |
| Max replan cycles per episode | 2 | `safety.max_replans` |
| Max action steps per subtask | 200 | `safety.max_steps_per_subtask` |
| Max total action steps per episode | 1000 | `safety.max_steps_per_episode` |
| Timeout per subtask (seconds) | 60 | `safety.timeout_subtask_s` |
| Timeout per episode (seconds) | 300 | `safety.timeout_episode_s` |

Exceeding any limit triggers a `SAFETY_LIMIT_EXCEEDED` terminal state.

### Action schema and range validation

All actions produced by `RobotPolicy.act()` are validated against a Pydantic schema before being passed to `RobotEnvironment.step()`. Validation failures trigger `INVALID_ACTION` and halt execution for that step. The executor does not retry invalid actions automatically — a retry must be explicitly authorised by the graph's recovery node.

### Human approval checkpoints

Operations classified as high-risk or ambiguous (e.g., the planner is uncertain, confidence is below threshold, or the task involves an unknown object) require human approval before execution. In mock and replay modes this is simulated by a configurable flag. In a future hardware deployment this must be a real blocking confirmation.

### Fail-closed behavior

When any of the following occur, the agent transitions to the `FAILED_SAFETY` terminal state and stops:

- Plan validation fails and max replans are exhausted
- Policy output fails schema validation more than once consecutively
- Safety gate returns `UNSAFE` for the current subtask
- Telemetry contains a credential or secret (redacted and logged; execution halted)

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
