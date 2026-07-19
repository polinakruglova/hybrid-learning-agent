import pytest

from src.agents import RuleAgent
from src.rules import Rule, RuleStore
from src.state import State


def make_state(
    x: int = 1,
    y: int = 1,
    has_key: bool = False,
    key_position: tuple[int, int] | None = None,
    door_position: tuple[int, int] | None = None,
    goal_position: tuple[int, int] | None = None,
) -> State:
    return State(
        agent_position=(x, y),
        agent_direction=0,
        has_key=has_key,
        key_position=key_position,
        door_position=door_position,
        goal_position=goal_position,
        key_visible=key_position is not None,
        door_visible=door_position is not None,
        goal_visible=goal_position is not None,
    )


def test_rule_agent_selects_matching_rule() -> None:
    rule = Rule(
        conditions={
            "has_key": False,
            "near_key": True,
        },
        action=2,
        support=10,
        successes=9,
        total_reward=8.0,
    )

    store = RuleStore([rule])
    agent = RuleAgent(store)

    state = make_state(
        x=1,
        y=2,
        key_position=(1, 1),
    )

    action = agent.choose_action(state)

    assert action == 2
    assert agent.last_rule is rule
    assert agent.last_action == 2
    assert agent.last_decision_source == "rule"


def test_rule_agent_selects_best_matching_rule() -> None:
    weak_rule = Rule(
        conditions={
            "near_goal": True,
        },
        action=1,
        support=10,
        successes=4,
        total_reward=2.0,
    )

    strong_rule = Rule(
        conditions={
            "near_goal": True,
        },
        action=2,
        support=10,
        successes=9,
        total_reward=8.0,
    )

    store = RuleStore(
        [
            weak_rule,
            strong_rule,
        ]
    )

    agent = RuleAgent(store)

    state = make_state(
        x=1,
        y=1,
        goal_position=(1, 2),
    )

    action = agent.choose_action(state)

    assert action == 2
    assert agent.last_rule is strong_rule


def test_rule_agent_returns_none_without_matching_rule() -> None:
    store = RuleStore(
        [
            Rule(
                conditions={"has_key": True},
                action=5,
            )
        ]
    )

    agent = RuleAgent(store)

    state = make_state(has_key=False)

    action = agent.choose_action(state)

    assert action is None
    assert agent.last_rule is None
    assert agent.last_decision_source == "none"


def test_rule_agent_uses_fallback_action() -> None:
    store = RuleStore()

    agent = RuleAgent(
        rule_store=store,
        fallback_action=0,
    )

    action = agent.choose_action(
        make_state()
    )

    assert action == 0
    assert agent.last_rule is None
    assert agent.last_action == 0
    assert agent.last_decision_source == "fallback_action"


def test_rule_agent_uses_fallback_policy() -> None:
    def fallback_policy(state: State) -> int:
        if state.has_key:
            return 5

        return 1

    agent = RuleAgent(
        rule_store=RuleStore(),
        fallback_policy=fallback_policy,
    )

    action_without_key = agent.choose_action(
        make_state(has_key=False)
    )

    action_with_key = agent.choose_action(
        make_state(has_key=True)
    )

    assert action_without_key == 1
    assert action_with_key == 5
    assert agent.last_decision_source == "fallback_policy"


def test_fallback_policy_has_priority_over_fallback_action() -> None:
    def fallback_policy(state: State) -> int:
        return 3

    agent = RuleAgent(
        rule_store=RuleStore(),
        fallback_action=0,
        fallback_policy=fallback_policy,
    )

    action = agent.choose_action(
        make_state()
    )

    assert action == 3
    assert agent.last_decision_source == "fallback_policy"


def test_rule_agent_finds_all_matching_rules() -> None:
    first_rule = Rule(
        conditions={"near_key": True},
        action=1,
    )

    second_rule = Rule(
        conditions={
            "near_key": True,
            "has_key": False,
        },
        action=2,
    )

    non_matching_rule = Rule(
        conditions={"has_key": True},
        action=5,
    )

    agent = RuleAgent(
        RuleStore(
            [
                first_rule,
                second_rule,
                non_matching_rule,
            ]
        )
    )

    state = make_state(
        x=1,
        y=2,
        key_position=(1, 1),
        has_key=False,
    )

    matching = agent.find_matching_rules(state)

    assert matching == [
        first_rule,
        second_rule,
    ]


def test_rule_agent_explains_rule_decision() -> None:
    rule = Rule(
        conditions={"near_door": True},
        action=5,
        support=4,
        successes=3,
        total_reward=2.0,
    )

    agent = RuleAgent(
        RuleStore([rule])
    )

    state = make_state(
        x=1,
        y=1,
        door_position=(1, 2),
    )

    agent.choose_action(state)

    explanation = agent.explain_decision()

    assert explanation["action"] == 5
    assert explanation["source"] == "rule"
    assert explanation["rule"] is not None
    assert explanation["rule"]["action"] == 5
    assert explanation["predicates"]["near_door"] is True


def test_rule_agent_explains_fallback_decision() -> None:
    agent = RuleAgent(
        RuleStore(),
        fallback_action=0,
    )

    agent.choose_action(
        make_state()
    )

    explanation = agent.explain_decision()

    assert explanation["action"] == 0
    assert explanation["source"] == "fallback_action"
    assert explanation["rule"] is None
    assert explanation["predicates"] is not None


def test_rule_agent_resets_decision_history() -> None:
    agent = RuleAgent(
        RuleStore(),
        fallback_action=0,
    )

    agent.choose_action(
        make_state()
    )

    agent.reset_decision_history()

    assert agent.last_rule is None
    assert agent.last_predicates is None
    assert agent.last_action is None
    assert agent.last_decision_source is None


def test_rule_agent_rejects_invalid_store() -> None:
    with pytest.raises(TypeError):
        RuleAgent(
            rule_store="invalid",  # type: ignore[arg-type]
        )


def test_rule_agent_rejects_invalid_fallback_action() -> None:
    with pytest.raises(TypeError):
        RuleAgent(
            rule_store=RuleStore(),
            fallback_action="left",  # type: ignore[arg-type]
        )


def test_rule_agent_rejects_invalid_fallback_policy() -> None:
    with pytest.raises(TypeError):
        RuleAgent(
            rule_store=RuleStore(),
            fallback_policy=123,  # type: ignore[arg-type]
        )


def test_rule_agent_validates_fallback_policy_result() -> None:
    def invalid_policy(state: State) -> str:
        return "move"

    agent = RuleAgent(
        rule_store=RuleStore(),
        fallback_policy=invalid_policy,  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError):
        agent.choose_action(
            make_state()
        )