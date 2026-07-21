import pytest

from src.agents import HybridAgent, RuleAgent
from src.learning import LearningController
from src.memory import ExperienceBuffer
from src.predicates import PredicateGenerator
from src.rl import QTablePolicy
from src.rules import RuleStore
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


def make_controller() -> LearningController:
    rule_store = RuleStore()

    predicate_generator = PredicateGenerator()

    rule_agent = RuleAgent(
        rule_store=rule_store,
        predicate_generator=predicate_generator,
    )

    q_policy = QTablePolicy(
        actions=[0, 1, 2, 3, 4, 5, 6],
        learning_rate=0.5,
        discount_factor=0.9,
        epsilon=0.0,
        seed=42,
    )

    hybrid_agent = HybridAgent(
        rule_agent=rule_agent,
        rl_policy=q_policy,
    )

    experience_buffer = ExperienceBuffer(
        capacity=100,
        seed=42,
    )

    return LearningController(
        agent=hybrid_agent,
        q_policy=q_policy,
        experience_buffer=experience_buffer,
        rule_store=rule_store,
    )


def test_controller_creation() -> None:
    controller = make_controller()

    assert controller.total_steps == 0
    assert controller.total_episodes == 0
    assert controller.successful_episodes == 0
    assert controller.total_reward == 0.0
    assert controller.success_rate == 0.0
    assert controller.average_reward_per_step == 0.0

    assert len(controller.experience_buffer) == 0
    assert len(controller.rule_store) == 0
    assert controller.q_policy.state_count == 0


def test_controller_chooses_action() -> None:
    controller = make_controller()

    state = make_state()

    action = controller.choose_action(state)

    assert action in controller.q_policy.actions
    assert controller.agent.last_action == action


def test_controller_learns_transition() -> None:
    controller = make_controller()

    state = make_state(x=1)
    next_state = make_state(x=2)

    experience = controller.learn(
        state=state,
        action=1,
        reward=2.0,
        next_state=next_state,
        done=False,
        success=False,
        source="rl",
    )

    assert experience.state == state
    assert experience.action == 1
    assert experience.reward == 2.0
    assert experience.next_state == next_state
    assert experience.done is False
    assert experience.success is False
    assert experience.source == "rl"

    assert len(controller.experience_buffer) == 1
    assert controller.total_steps == 1
    assert controller.total_reward == 2.0

    assert controller.last_experience is experience
    assert controller.last_q_value is not None


def test_controller_updates_q_table() -> None:
    controller = make_controller()

    state = make_state(x=1)
    next_state = make_state(x=2)

    controller.learn(
        state=state,
        action=1,
        reward=2.0,
        next_state=next_state,
        done=True,
        source="rl",
    )

    q_value = controller.q_policy.get_q_value(
        state,
        action=1,
    )

    # new_q = 0 + 0.5 * (2 - 0)
    assert q_value == pytest.approx(1.0)


def test_controller_uses_default_agent_source() -> None:
    controller = make_controller()

    state = make_state()
    next_state = make_state(x=2)

    action = controller.choose_action(state)

    experience = controller.learn(
        state=state,
        action=action,
        reward=1.0,
        next_state=next_state,
    )

    assert experience.source == "rl"


def test_controller_stores_experience_in_buffer() -> None:
    controller = make_controller()

    state = make_state(x=1)
    next_state = make_state(x=2)

    experience = controller.learn(
        state=state,
        action=1,
        reward=1.0,
        next_state=next_state,
        source="agent",
    )

    stored_experiences = (
        controller.experience_buffer.experiences
    )

    assert len(stored_experiences) == 1
    assert stored_experiences[0] is experience


def test_controller_does_not_generate_rules_during_learn() -> None:
    controller = make_controller()

    controller.learn(
        state=make_state(
            x=1,
            key_position=(2, 1),
        ),
        action=1,
        reward=5.0,
        next_state=make_state(
            x=2,
            key_position=(2, 1),
        ),
        done=True,
        success=True,
        source="agent",
    )

    # LearningController теперь только:
    # 1. сохраняет опыт;
    # 2. обновляет Q-таблицу.
    #
    # Символические правила создаются отдельно
    # через NeuralRuleTrainer.
    assert len(controller.rule_store) == 0
    assert len(controller.experience_buffer) == 1


def test_controller_step_uses_last_action() -> None:
    controller = make_controller()

    state = make_state()
    next_state = make_state(x=2)

    action = controller.choose_action(state)

    experience = controller.step(
        state=state,
        reward=1.0,
        next_state=next_state,
    )

    assert experience.action == action
    assert experience.source == "rl"
    assert controller.total_steps == 1


def test_controller_step_requires_previous_action() -> None:
    controller = make_controller()

    with pytest.raises(RuntimeError):
        controller.step(
            state=make_state(),
            reward=1.0,
            next_state=make_state(x=2),
        )


def test_controller_counts_completed_episode() -> None:
    controller = make_controller()

    controller.learn(
        state=make_state(),
        action=1,
        reward=1.0,
        next_state=make_state(x=2),
        done=True,
        success=False,
        source="rl",
    )

    assert controller.total_episodes == 1
    assert controller.successful_episodes == 0
    assert controller.success_rate == 0.0


def test_controller_counts_successful_episode() -> None:
    controller = make_controller()

    controller.learn(
        state=make_state(),
        action=1,
        reward=5.0,
        next_state=make_state(x=2),
        done=True,
        success=True,
        source="rl",
    )

    assert controller.total_episodes == 1
    assert controller.successful_episodes == 1
    assert controller.success_rate == 1.0


def test_controller_calculates_statistics() -> None:
    controller = make_controller()

    controller.learn(
        state=make_state(x=1),
        action=1,
        reward=2.0,
        next_state=make_state(x=2),
        source="rl",
    )

    controller.learn(
        state=make_state(x=2),
        action=2,
        reward=4.0,
        next_state=make_state(x=3),
        done=True,
        success=True,
        source="rl",
    )

    statistics = controller.statistics()

    assert statistics["total_steps"] == 2
    assert statistics["total_episodes"] == 1
    assert statistics["successful_episodes"] == 1
    assert statistics["success_rate"] == 1.0
    assert statistics["total_reward"] == 6.0

    assert statistics[
        "average_reward_per_step"
    ] == pytest.approx(3.0)

    assert statistics["experience_count"] == 2
    assert statistics["q_state_count"] >= 2


def test_controller_decays_epsilon() -> None:
    controller = make_controller()

    controller.q_policy.epsilon = 1.0

    new_epsilon = controller.decay_epsilon(
        decay_rate=0.5,
        minimum_epsilon=0.1,
    )

    assert new_epsilon == pytest.approx(0.5)
    assert controller.q_policy.epsilon == pytest.approx(
        0.5
    )


def test_controller_resets_statistics_only() -> None:
    controller = make_controller()

    controller.learn(
        state=make_state(),
        action=1,
        reward=2.0,
        next_state=make_state(x=2),
        source="rl",
    )

    controller.reset_statistics()

    assert controller.total_steps == 0
    assert controller.total_episodes == 0
    assert controller.successful_episodes == 0
    assert controller.total_reward == 0.0

    assert controller.last_experience is None
    assert controller.last_q_value is None

    # Обучающие данные остаются.
    assert len(controller.experience_buffer) == 1
    assert controller.q_policy.state_count > 0


def test_controller_clears_learning_data() -> None:
    controller = make_controller()

    controller.learn(
        state=make_state(),
        action=1,
        reward=2.0,
        next_state=make_state(x=2),
        source="rl",
    )

    controller.clear_learning_data()

    assert controller.total_steps == 0
    assert controller.total_episodes == 0
    assert controller.successful_episodes == 0
    assert controller.total_reward == 0.0

    assert len(controller.experience_buffer) == 0
    assert controller.q_policy.state_count == 0
    assert len(controller.rule_store) == 0

    assert controller.last_experience is None
    assert controller.last_q_value is None
    assert controller.agent.last_action is None


def test_controller_rejects_invalid_transition() -> None:
    controller = make_controller()

    with pytest.raises(TypeError):
        controller.learn(
            state="invalid",  # type: ignore[arg-type]
            action=1,
            reward=1.0,
            next_state=make_state(),
        )

    with pytest.raises(TypeError):
        controller.learn(
            state=make_state(),
            action="move",  # type: ignore[arg-type]
            reward=1.0,
            next_state=make_state(),
        )

    with pytest.raises(TypeError):
        controller.learn(
            state=make_state(),
            action=1,
            reward="high",  # type: ignore[arg-type]
            next_state=make_state(),
        )

    with pytest.raises(TypeError):
        controller.learn(
            state=make_state(),
            action=1,
            reward=1.0,
            next_state="invalid",  # type: ignore[arg-type]
        )


def test_controller_rejects_invalid_done() -> None:
    controller = make_controller()

    with pytest.raises(TypeError):
        controller.learn(
            state=make_state(),
            action=1,
            reward=1.0,
            next_state=make_state(x=2),
            done=1,  # type: ignore[arg-type]
        )


def test_controller_rejects_invalid_success() -> None:
    controller = make_controller()

    with pytest.raises(TypeError):
        controller.learn(
            state=make_state(),
            action=1,
            reward=1.0,
            next_state=make_state(x=2),
            success=1,  # type: ignore[arg-type]
        )


def test_controller_rejects_empty_source() -> None:
    controller = make_controller()

    with pytest.raises(ValueError):
        controller.learn(
            state=make_state(),
            action=1,
            reward=1.0,
            next_state=make_state(x=2),
            source="",
        )


def test_controller_rejects_whitespace_source() -> None:
    controller = make_controller()

    with pytest.raises(ValueError):
        controller.learn(
            state=make_state(),
            action=1,
            reward=1.0,
            next_state=make_state(x=2),
            source="   ",
        )