"""SafetyChecker — software-layer gate applied before each subtask execution.

This is a software guard (keyword allowlist / blocklist) and does NOT provide
physical safety guarantees. Physical guarantees require hardware validation
(joint limits, emergency stops, collision detection) which is out of scope
until a real robot is available (see docs/safety.md).
"""

_DEFAULT_ALLOWED: frozenset[str] = frozenset(
    [
        "approach",
        "grasp",
        "grip",
        "pick",
        "grab",
        "take",
        "lift",
        "raise",
        "place",
        "put",
        "set",
        "release",
        "move",
        "push",
        "pull",
        "open",
        "close",
        "lower",
        "drop",
    ]
)

_DEFAULT_BLOCKED: frozenset[str] = frozenset(
    [
        "human",
        "person",
        "face",
        "eye",
        "sharp",
        "blade",
        "knife",
        "hot",
        "fire",
        "flame",
        "weapon",
        "explosive",
        "harmful",
        "dangerous",
        "child",
    ]
)


class SafetyChecker:
    """Validates subtask instructions against an allowlist of manipulation verbs
    and a blocklist of hazardous terms.

    Parameters
    ----------
    allowed_verbs:
        Frozenset of lowercase terms; at least one must appear in the instruction.
    blocked_terms:
        Frozenset of lowercase terms; any match causes rejection.
    """

    def __init__(
        self,
        allowed_verbs: frozenset[str] = _DEFAULT_ALLOWED,
        blocked_terms: frozenset[str] = _DEFAULT_BLOCKED,
    ) -> None:
        self._allowed = allowed_verbs
        self._blocked = blocked_terms

    def check(self, instruction: str) -> tuple[bool, str]:
        """Return (ok, reason). reason is empty when ok=True.

        A subtask passes when:
        1. No blocked term appears in the instruction.
        2. At least one allowed manipulation verb appears.
        """
        lower = instruction.lower()

        for term in self._blocked:
            if term in lower:
                return False, f"blocked term detected: {term!r}"

        if not any(verb in lower for verb in self._allowed):
            return False, (
                "instruction contains no recognised manipulation verb "
                f"(allowed: {sorted(self._allowed)[:5]}…)"
            )

        return True, ""
