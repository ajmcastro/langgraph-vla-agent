"""Unit tests for SafetyChecker."""

from langgraph_vla_agent.agent.safety import SafetyChecker


def test_pick_instruction_is_allowed() -> None:
    ok, reason = SafetyChecker().check("pick up the cube")
    assert ok
    assert reason == ""


def test_place_instruction_is_allowed() -> None:
    ok, _reason = SafetyChecker().check("place the object at the target")
    assert ok


def test_approach_instruction_is_allowed() -> None:
    ok, _reason = SafetyChecker().check("approach the object slowly")
    assert ok


def test_grasp_instruction_is_allowed() -> None:
    ok, _reason = SafetyChecker().check("grasp the block firmly")
    assert ok


def test_blocked_keyword_human_is_rejected() -> None:
    ok, reason = SafetyChecker().check("move toward the human")
    assert not ok
    assert "human" in reason


def test_blocked_keyword_sharp_is_rejected() -> None:
    ok, _reason = SafetyChecker().check("grasp the sharp knife")
    assert not ok


def test_instruction_with_no_allowed_verb_is_rejected() -> None:
    ok, reason = SafetyChecker().check("illuminate the scene")
    assert not ok
    assert "manipulation verb" in reason


def test_case_insensitive_blocked_check() -> None:
    ok, _ = SafetyChecker().check("Move toward the HUMAN")
    assert not ok


def test_case_insensitive_allowed_check() -> None:
    ok, _ = SafetyChecker().check("PICK UP the block")
    assert ok


def test_custom_allowed_verbs_respected() -> None:
    checker = SafetyChecker(allowed_verbs=frozenset(["rotate"]))
    ok, _ = checker.check("rotate the joint")
    assert ok


def test_custom_allowed_verbs_rejects_unknown_verb() -> None:
    checker = SafetyChecker(allowed_verbs=frozenset(["rotate"]))
    ok, _ = checker.check("pick up the block")
    assert not ok  # "pick" not in custom allowed set


def test_custom_blocked_terms_respected() -> None:
    checker = SafetyChecker(blocked_terms=frozenset(["coffee"]))
    ok, _ = checker.check("pick up the coffee mug")
    assert not ok


def test_empty_instruction_is_rejected() -> None:
    ok, _ = SafetyChecker().check("")
    assert not ok


def test_blocked_takes_priority_over_allowed() -> None:
    # "human" is blocked even though "move" is allowed.
    ok, reason = SafetyChecker().check("move the human mannequin")
    assert not ok
    assert "human" in reason
