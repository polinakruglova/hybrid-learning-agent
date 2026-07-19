import pytest

from src.rules import Rule


def test_rule_matches_predicates() -> None:
    rule = Rule(
        conditions={
            "key_visible": True,
            "has_key": False,
            "key_above": True,
        },
        action=2,
    )

    predicates = {
        "key_visible": True,
        "has_key": False,
        "key_above": True,
        "door_locked": True,
    }

    assert rule.matches(predicates) is True


def test_rule_does_not_match_different_value() -> None:
    rule = Rule(
        conditions={
            "has_key": False,
            "near_key": True,
        },
        action=3,
    )

    predicates = {
        "has_key": True,
        "near_key": True,
    }

    assert rule.matches(predicates) is False


def test_rule_does_not_match_missing_predicate() -> None:
    rule = Rule(
        conditions={
            "near_door": True,
            "door_locked": True,
        },
        action=5,
    )

    predicates = {
        "near_door": True,
    }

    assert rule.matches(predicates) is False


def test_rule_updates_statistics_after_success() -> None:
    rule = Rule(
        conditions={"near_goal": True},
        action=2,
    )

    rule.update(
        reward=1.0,
        success=True,
    )

    assert rule.support == 1
    assert rule.successes == 1
    assert rule.total_reward == 1.0
    assert rule.confidence == 1.0
    assert rule.average_reward == 1.0


def test_rule_updates_statistics_after_failure() -> None:
    rule = Rule(
        conditions={"near_door": True},
        action=4,
    )

    rule.update(
        reward=-0.2,
        success=False,
    )

    assert rule.support == 1
    assert rule.successes == 0
    assert rule.confidence == 0.0
    assert rule.average_reward == pytest.approx(-0.2)


def test_rule_calculates_statistics_for_multiple_updates() -> None:
    rule = Rule(
        conditions={"goal_visible": True},
        action=1,
    )

    rule.update(reward=1.0, success=True)
    rule.update(reward=0.5, success=True)
    rule.update(reward=-0.5, success=False)

    assert rule.support == 3
    assert rule.successes == 2
    assert rule.confidence == pytest.approx(2 / 3)
    assert rule.average_reward == pytest.approx(1 / 3)


def test_rule_specificity() -> None:
    rule = Rule(
        conditions={
            "door_visible": True,
            "door_locked": True,
            "has_key": True,
        },
        action=5,
    )

    assert rule.specificity == 3


def test_rule_rejects_empty_conditions() -> None:
    with pytest.raises(ValueError):
        Rule(
            conditions={},
            action=0,
        )


def test_rule_rejects_invalid_statistics() -> None:
    with pytest.raises(ValueError):
        Rule(
            conditions={"has_key": True},
            action=0,
            support=1,
            successes=2,
        )


def test_rule_to_dict() -> None:
    rule = Rule(
        conditions={"near_key": True},
        action=3,
        support=2,
        successes=1,
        total_reward=0.5,
        metadata={"source": "teacher"},
    )

    data = rule.to_dict()

    assert data["conditions"] == {"near_key": True}
    assert data["action"] == 3
    assert data["confidence"] == 0.5
    assert data["average_reward"] == 0.25
    assert data["metadata"]["source"] == "teacher"