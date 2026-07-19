from src.memory import Experience
from src.rules import RuleMiner, RuleStore
from src.state import State


def make_state(
    x: int = 1,
    y: int = 1,
    direction: int = 0,
    has_key: bool = False,
    key_position: tuple[int, int] | None = None,
    door_position: tuple[int, int] | None = None,
    goal_position: tuple[int, int] | None = None,
) -> State:
    return State(
        agent_position=(x, y),
        agent_direction=direction,
        has_key=has_key,
        key_position=key_position,
        door_position=door_position,
        goal_position=goal_position,
        key_visible=key_position is not None,
        door_visible=door_position is not None,
        goal_visible=goal_position is not None,
    )


def make_successful_experience() -> Experience:
    return Experience(
        state=make_state(
            x=1,
            y=3,
            key_position=(1, 2),
        ),
        action=2,
        reward=1.0,
        next_state=make_state(
            x=1,
            y=2,
            key_position=(1, 2),
        ),
        success=True,
        source="teacher",
    )


def test_miner_accepts_successful_positive_experience() -> None:
    miner = RuleMiner()

    experience = make_successful_experience()

    assert miner.can_mine(experience) is True


def test_miner_rejects_unsuccessful_experience() -> None:
    miner = RuleMiner()

    experience = Experience(
        state=make_state(),
        action=2,
        reward=1.0,
        next_state=make_state(x=2),
        success=False,
    )

    assert miner.can_mine(experience) is False
    assert miner.mine(experience) is None


def test_miner_rejects_non_positive_reward() -> None:
    miner = RuleMiner()

    experience = Experience(
        state=make_state(),
        action=2,
        reward=0.0,
        next_state=make_state(x=2),
        success=True,
    )

    assert miner.can_mine(experience) is False
    assert miner.mine(experience) is None


def test_miner_respects_allowed_sources() -> None:
    miner = RuleMiner(
        allowed_sources={"teacher"},
    )

    teacher_experience = make_successful_experience()

    agent_experience = Experience(
        state=make_state(),
        action=2,
        reward=1.0,
        next_state=make_state(x=2),
        success=True,
        source="agent",
    )

    assert miner.can_mine(teacher_experience) is True
    assert miner.can_mine(agent_experience) is False


def test_miner_creates_rule_from_experience() -> None:
    miner = RuleMiner()

    rule = miner.mine(
        make_successful_experience()
    )

    assert rule is not None
    assert rule.action == 2
    assert rule.support == 1
    assert rule.successes == 1
    assert rule.total_reward == 1.0
    assert rule.metadata["source"] == "teacher"
    assert rule.metadata["mined"] is True


def test_miner_uses_state_predicates_as_conditions() -> None:
    miner = RuleMiner(
        selected_predicates=[
            "has_key",
            "key_visible",
            "near_key",
            "key_above",
            "same_column_with_key",
        ]
    )

    rule = miner.mine(
        make_successful_experience()
    )

    assert rule is not None

    assert rule.conditions == {
        "has_key": False,
        "key_visible": True,
        "near_key": True,
        "key_above": True,
        "same_column_with_key": True,
    }


def test_miner_ignores_unknown_predicates() -> None:
    miner = RuleMiner(
        selected_predicates=[
            "distance_to_goal",
            "goal_visible",
        ]
    )

    experience = Experience(
        state=make_state(),
        action=2,
        reward=1.0,
        next_state=make_state(x=2),
        success=True,
    )

    rule = miner.mine(experience)

    assert rule is not None
    assert rule.conditions == {
        "goal_visible": False,
    }


def test_miner_returns_none_when_no_conditions_remain() -> None:
    miner = RuleMiner(
        selected_predicates=[
            "distance_to_goal",
        ]
    )

    experience = Experience(
        state=make_state(),
        action=2,
        reward=1.0,
        next_state=make_state(x=2),
        success=True,
    )

    assert miner.mine(experience) is None


def test_miner_creates_multiple_rules() -> None:
    miner = RuleMiner(
        selected_predicates=["has_key"]
    )

    experiences = [
        make_successful_experience(),
        Experience(
            state=make_state(has_key=True),
            action=5,
            reward=0.5,
            next_state=make_state(has_key=True),
            success=True,
        ),
    ]

    rules = miner.mine_many(experiences)

    assert len(rules) == 2
    assert rules[0].action == 2
    assert rules[1].action == 5


def test_miner_adds_rules_to_store() -> None:
    miner = RuleMiner(
        selected_predicates=[
            "has_key",
            "near_key",
        ]
    )

    store = RuleStore()

    added = miner.mine_into_store(
        [make_successful_experience()],
        store,
    )

    assert len(added) == 1
    assert len(store) == 1
    assert store.rules[0] is added[0]


def test_miner_does_not_add_duplicate_rule() -> None:
    miner = RuleMiner(
        selected_predicates=[
            "has_key",
            "near_key",
        ]
    )

    experience = make_successful_experience()
    store = RuleStore()

    first_added = miner.mine_into_store(
        [experience],
        store,
    )

    second_added = miner.mine_into_store(
        [experience],
        store,
    )

    assert len(first_added) == 1
    assert second_added == []
    assert len(store) == 1


def test_miner_applies_minimum_reward_threshold() -> None:
    miner = RuleMiner(
        minimum_reward=0.5,
    )

    weak_experience = Experience(
        state=make_state(),
        action=2,
        reward=0.5,
        next_state=make_state(x=2),
        success=True,
    )

    strong_experience = Experience(
        state=make_state(),
        action=2,
        reward=0.6,
        next_state=make_state(x=2),
        success=True,
    )

    assert miner.mine(weak_experience) is None
    assert miner.mine(strong_experience) is not None