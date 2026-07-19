import pytest

from src.rules import Rule, RuleStore


def test_store_starts_empty() -> None:
    store = RuleStore()

    assert len(store) == 0
    assert store.rules == ()


def test_store_accepts_initial_rules() -> None:
    rule = Rule(
        conditions={"near_key": True},
        action=3,
    )

    store = RuleStore([rule])

    assert len(store) == 1
    assert store.rules[0] is rule


def test_store_adds_rule() -> None:
    store = RuleStore()

    rule = Rule(
        conditions={"has_key": True},
        action=2,
    )

    store.add(rule)

    assert len(store) == 1
    assert store.rules[0] is rule


def test_store_does_not_add_equivalent_duplicate() -> None:
    store = RuleStore()

    first = Rule(
        conditions={"near_door": True},
        action=5,
    )

    duplicate = Rule(
        conditions={"near_door": True},
        action=5,
        support=10,
        successes=8,
    )

    store.add(first)
    store.add(duplicate)

    assert len(store) == 1
    assert store.rules[0] is first


def test_store_allows_same_conditions_with_different_action() -> None:
    store = RuleStore()

    store.add(
        Rule(
            conditions={"near_door": True},
            action=2,
        )
    )

    store.add(
        Rule(
            conditions={"near_door": True},
            action=5,
        )
    )

    assert len(store) == 2


def test_store_finds_matching_rules() -> None:
    matching_rule = Rule(
        conditions={
            "near_key": True,
            "has_key": False,
        },
        action=3,
    )

    non_matching_rule = Rule(
        conditions={
            "near_door": True,
            "has_key": True,
        },
        action=5,
    )

    store = RuleStore(
        [
            matching_rule,
            non_matching_rule,
        ]
    )

    predicates = {
        "near_key": True,
        "has_key": False,
        "near_door": False,
    }

    matching = store.find_matching(predicates)

    assert matching == [matching_rule]


def test_store_returns_none_when_no_rule_matches() -> None:
    store = RuleStore(
        [
            Rule(
                conditions={"has_key": True},
                action=2,
            )
        ]
    )

    best = store.select_best(
        {"has_key": False}
    )

    assert best is None


def test_store_selects_rule_with_best_confidence() -> None:
    weak_rule = Rule(
        conditions={"goal_visible": True},
        action=0,
        support=10,
        successes=5,
        total_reward=5.0,
    )

    strong_rule = Rule(
        conditions={"goal_visible": True},
        action=1,
        support=10,
        successes=9,
        total_reward=2.0,
    )

    store = RuleStore(
        [
            weak_rule,
            strong_rule,
        ]
    )

    best = store.select_best(
        {"goal_visible": True}
    )

    assert best is strong_rule


def test_store_uses_reward_as_second_priority() -> None:
    lower_reward = Rule(
        conditions={"door_visible": True},
        action=0,
        support=10,
        successes=8,
        total_reward=2.0,
    )

    higher_reward = Rule(
        conditions={"door_visible": True},
        action=1,
        support=10,
        successes=8,
        total_reward=7.0,
    )

    store = RuleStore(
        [
            lower_reward,
            higher_reward,
        ]
    )

    best = store.select_best(
        {"door_visible": True}
    )

    assert best is higher_reward


def test_store_uses_specificity_as_third_priority() -> None:
    general_rule = Rule(
        conditions={"door_visible": True},
        action=0,
        support=10,
        successes=8,
        total_reward=5.0,
    )

    specific_rule = Rule(
        conditions={
            "door_visible": True,
            "door_locked": True,
        },
        action=5,
        support=10,
        successes=8,
        total_reward=5.0,
    )

    store = RuleStore(
        [
            general_rule,
            specific_rule,
        ]
    )

    best = store.select_best(
        {
            "door_visible": True,
            "door_locked": True,
        }
    )

    assert best is specific_rule


def test_store_removes_rule() -> None:
    rule = Rule(
        conditions={"near_goal": True},
        action=2,
    )

    store = RuleStore([rule])

    result = store.remove(rule)

    assert result is True
    assert len(store) == 0


def test_store_returns_false_for_missing_rule() -> None:
    store = RuleStore()

    rule = Rule(
        conditions={"near_goal": True},
        action=2,
    )

    assert store.remove(rule) is False


def test_store_prunes_weak_rule() -> None:
    weak_rule = Rule(
        conditions={"near_key": True},
        action=1,
        support=20,
        successes=2,
        total_reward=-5.0,
    )

    strong_rule = Rule(
        conditions={"near_goal": True},
        action=2,
        support=20,
        successes=18,
        total_reward=15.0,
    )

    store = RuleStore(
        [
            weak_rule,
            strong_rule,
        ]
    )

    removed = store.prune(
        minimum_support=10,
        minimum_confidence=0.2,
    )

    assert removed == [weak_rule]
    assert store.rules == (strong_rule,)


def test_store_does_not_prune_new_rule() -> None:
    new_rule = Rule(
        conditions={"near_key": True},
        action=1,
        support=2,
        successes=0,
        total_reward=-1.0,
    )

    store = RuleStore([new_rule])

    removed = store.prune(
        minimum_support=10,
        minimum_confidence=0.5,
    )

    assert removed == []
    assert store.rules == (new_rule,)


def test_store_prunes_by_average_reward() -> None:
    negative_rule = Rule(
        conditions={"door_locked": True},
        action=5,
        support=10,
        successes=9,
        total_reward=-5.0,
    )

    store = RuleStore([negative_rule])

    removed = store.prune(
        minimum_support=5,
        minimum_confidence=0.5,
        minimum_average_reward=0.0,
    )

    assert removed == [negative_rule]
    assert len(store) == 0


def test_store_rejects_invalid_prune_parameters() -> None:
    store = RuleStore()

    with pytest.raises(ValueError):
        store.prune(
            minimum_support=-1,
        )

    with pytest.raises(ValueError):
        store.prune(
            minimum_confidence=1.5,
        )


def test_store_converts_rules_to_list() -> None:
    rule = Rule(
        conditions={"goal_visible": True},
        action=2,
        support=2,
        successes=1,
        total_reward=0.5,
    )

    store = RuleStore([rule])

    data = store.to_list()

    assert len(data) == 1
    assert data[0]["conditions"] == {
        "goal_visible": True
    }
    assert data[0]["action"] == 2
    assert data[0]["confidence"] == 0.5