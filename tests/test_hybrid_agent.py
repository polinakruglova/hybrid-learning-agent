import pytest

from src.agents import HybridAgent, RuleAgent
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


def test_hybrid_agent_uses_rule_when_available() -> None:
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

    rule_agent = RuleAgent(
        RuleStore([rule])
    )

    rl_calls = 0

    def rl_policy(state: State) -> int:
        nonlocal rl_calls
        rl_calls += 1
        return 4

    agent = HybridAgent(
        rule_agent=rule_agent,
        rl_policy=rl_policy,
    )

    state = make_state(
        x=1,
        y=2,
        key_position=(1, 1),
    )

    action = agent.choose_action(state)

    assert action == 2
    assert agent.last_source == "rule"
    assert agent.last_rule is rule
    assert rl_calls == 0


def test_hybrid_agent_uses_rl_without_matching_rule() -> None:
    rule_agent = RuleAgent(
        RuleStore()
    )

    def rl_policy(state: State) -> int:
        return 3

    agent = HybridAgent(
        rule_agent=rule_agent,
        rl_policy=rl_policy,
    )

    action = agent.choose_action(
        make_state()
    )

    assert action == 3
    assert agent.last_source == "rl"
    assert agent.last_rule is None


def test_hybrid_agent_ignores_rule_agent_fallback_action() -> None:
    rule_agent = RuleAgent(
        RuleStore(),
        fallback_action=0,
    )

    def rl_policy(state: State) -> int:
        return 4

    agent = HybridAgent(
        rule_agent=rule_agent,
        rl_policy=rl_policy,
    )

    action = agent.choose_action(
        make_state()
    )

    assert action == 4
    assert agent.last_source == "rl"


def test_hybrid_agent_ignores_rule_agent_fallback_policy() -> None:
    def rule_fallback(state: State) -> int:
        return 1

    rule_agent = RuleAgent(
        RuleStore(),
        fallback_policy=rule_fallback,
    )

    def rl_policy(state: State) -> int:
        return 5

    agent = HybridAgent(
        rule_agent=rule_agent,
        rl_policy=rl_policy,
    )

    action = agent.choose_action(
        make_state()
    )

    assert action == 5
    assert agent.last_source == "rl"


def test_hybrid_agent_explains_rule_decision() -> None:
    rule = Rule(
        conditions={"near_door": True},
        action=5,
        support=5,
        successes=4,
        total_reward=3.0,
    )

    rule_agent = RuleAgent(
        RuleStore([rule])
    )

    agent = HybridAgent(
        rule_agent=rule_agent,
        rl_policy=lambda state: 0,
    )

    agent.choose_action(
        make_state(
            x=1,
            y=1,
            door_position=(1, 2),
        )
    )

    explanation = agent.explain_decision()

    assert explanation["action"] == 5
    assert explanation["source"] == "rule"
    assert explanation["rule"] is not None
    assert explanation["rule"]["action"] == 5
    assert explanation["predicates"]["near_door"] is True
    assert explanation["state"]["agent_x"] == 1


def test_hybrid_agent_explains_rl_decision() -> None:
    rule_agent = RuleAgent(
        RuleStore()
    )

    agent = HybridAgent(
        rule_agent=rule_agent,
        rl_policy=lambda state: 3,
    )

    agent.choose_action(
        make_state(x=2, y=3)
    )

    explanation = agent.explain_decision()

    assert explanation["action"] == 3
    assert explanation["source"] == "rl"
    assert explanation["rule"] is None
    assert explanation["state"]["agent_x"] == 2
    assert explanation["state"]["agent_y"] == 3
    assert explanation["predicates"] is not None


def test_hybrid_agent_resets_history() -> None:
    agent = HybridAgent(
        rule_agent=RuleAgent(
            RuleStore()
        ),
        rl_policy=lambda state: 2,
    )

    agent.choose_action(
        make_state()
    )

    agent.reset_decision_history()

    assert agent.last_action is None
    assert agent.last_source is None
    assert agent.last_rule is None
    assert agent.last_state is None
    assert agent.rule_agent.last_action is None
    assert agent.rule_agent.last_predicates is None


def test_hybrid_agent_rejects_invalid_rule_agent() -> None:
    with pytest.raises(TypeError):
        HybridAgent(
            rule_agent="invalid",  # type: ignore[arg-type]
            rl_policy=lambda state: 0,
        )


def test_hybrid_agent_rejects_invalid_rl_policy() -> None:
    with pytest.raises(TypeError):
        HybridAgent(
            rule_agent=RuleAgent(
                RuleStore()
            ),
            rl_policy=123,  # type: ignore[arg-type]
        )


def test_hybrid_agent_rejects_invalid_state() -> None:
    agent = HybridAgent(
        rule_agent=RuleAgent(
            RuleStore()
        ),
        rl_policy=lambda state: 0,
    )

    with pytest.raises(TypeError):
        agent.choose_action(
            "invalid",  # type: ignore[arg-type]
        )


def test_hybrid_agent_validates_rl_action() -> None:
    def invalid_policy(state: State) -> str:
        return "move"

    agent = HybridAgent(
        rule_agent=RuleAgent(
            RuleStore()
        ),
        rl_policy=invalid_policy,  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError):
        agent.choose_action(
            make_state()
        )