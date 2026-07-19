import pytest

from src.rl import QTablePolicy
from src.state import State


def make_state(
    x: int = 1,
    y: int = 1,
    has_key: bool = False,
) -> State:
    return State(
        agent_position=(x, y),
        agent_direction=0,
        has_key=has_key,
    )


def test_q_table_creation() -> None:
    policy = QTablePolicy(
        actions=[0, 1, 2],
        learning_rate=0.2,
        discount_factor=0.9,
        epsilon=0.1,
    )

    assert policy.actions == (0, 1, 2)
    assert policy.learning_rate == 0.2
    assert policy.discount_factor == 0.9
    assert policy.epsilon == 0.1
    assert policy.state_count == 0


def test_q_table_initializes_unknown_state() -> None:
    policy = QTablePolicy(
        actions=[0, 1, 2]
    )

    state = make_state()

    values = policy.q_values(state)

    assert values == {
        0: 0.0,
        1: 0.0,
        2: 0.0,
    }

    assert policy.state_count == 1


def test_q_table_gets_and_sets_value() -> None:
    policy = QTablePolicy(
        actions=[0, 1, 2]
    )

    state = make_state()

    policy.set_q_value(
        state=state,
        action=1,
        value=2.5,
    )

    assert policy.get_q_value(state, 1) == 2.5


def test_q_values_returns_copy() -> None:
    policy = QTablePolicy(
        actions=[0, 1]
    )

    state = make_state()

    values = policy.q_values(state)
    values[0] = 999.0

    assert policy.get_q_value(state, 0) == 0.0


def test_best_actions_returns_all_tied_actions() -> None:
    policy = QTablePolicy(
        actions=[0, 1, 2]
    )

    state = make_state()

    policy.set_q_value(state, 0, 1.0)
    policy.set_q_value(state, 1, 3.0)
    policy.set_q_value(state, 2, 3.0)

    best = policy.best_actions(state)

    assert best == [1, 2]


def test_best_action_returns_highest_value_action() -> None:
    policy = QTablePolicy(
        actions=[0, 1, 2],
        seed=42,
    )

    state = make_state()

    policy.set_q_value(state, 0, 1.0)
    policy.set_q_value(state, 1, 5.0)
    policy.set_q_value(state, 2, 2.0)

    assert policy.best_action(state) == 1


def test_choose_action_without_exploration() -> None:
    policy = QTablePolicy(
        actions=[0, 1, 2],
        epsilon=1.0,
        seed=42,
    )

    state = make_state()

    policy.set_q_value(state, 0, 1.0)
    policy.set_q_value(state, 1, 5.0)
    policy.set_q_value(state, 2, 2.0)

    action = policy.choose_action(
        state,
        explore=False,
    )

    assert action == 1


def test_choose_action_with_zero_epsilon_uses_best_action() -> None:
    policy = QTablePolicy(
        actions=[0, 1, 2],
        epsilon=0.0,
        seed=42,
    )

    state = make_state()

    policy.set_q_value(state, 0, 1.0)
    policy.set_q_value(state, 1, 5.0)
    policy.set_q_value(state, 2, 2.0)

    assert policy.choose_action(state) == 1


def test_policy_can_be_called_as_function() -> None:
    policy = QTablePolicy(
        actions=[0, 1],
        epsilon=0.0,
    )

    state = make_state()

    policy.set_q_value(state, 1, 3.0)

    action = policy(state)

    assert action == 1


def test_q_learning_update_without_terminal_state() -> None:
    policy = QTablePolicy(
        actions=[0, 1],
        learning_rate=0.5,
        discount_factor=0.9,
        epsilon=0.0,
    )

    state = make_state(x=1)
    next_state = make_state(x=2)

    policy.set_q_value(
        next_state,
        action=0,
        value=2.0,
    )

    policy.set_q_value(
        next_state,
        action=1,
        value=4.0,
    )

    new_q = policy.update(
        state=state,
        action=0,
        reward=1.0,
        next_state=next_state,
        done=False,
    )

    # target = 1 + 0.9 * 4 = 4.6
    # new_q = 0 + 0.5 * (4.6 - 0) = 2.3

    assert new_q == pytest.approx(2.3)
    assert policy.get_q_value(state, 0) == pytest.approx(2.3)


def test_q_learning_update_terminal_state() -> None:
    policy = QTablePolicy(
        actions=[0, 1],
        learning_rate=0.5,
        discount_factor=0.9,
    )

    state = make_state(x=1)
    next_state = make_state(x=2)

    policy.set_q_value(
        next_state,
        action=1,
        value=100.0,
    )

    new_q = policy.update(
        state=state,
        action=0,
        reward=1.0,
        next_state=next_state,
        done=True,
    )

    # Для завершённого эпизода:
    # target = reward = 1
    # new_q = 0 + 0.5 * 1 = 0.5

    assert new_q == pytest.approx(0.5)


def test_q_learning_updates_existing_value() -> None:
    policy = QTablePolicy(
        actions=[0, 1],
        learning_rate=0.5,
        discount_factor=0.0,
    )

    state = make_state()
    next_state = make_state(x=2)

    policy.set_q_value(
        state,
        action=0,
        value=2.0,
    )

    new_q = policy.update(
        state=state,
        action=0,
        reward=4.0,
        next_state=next_state,
    )

    # new_q = 2 + 0.5 * (4 - 2) = 3

    assert new_q == pytest.approx(3.0)


def test_epsilon_can_be_changed() -> None:
    policy = QTablePolicy(
        actions=[0, 1],
        epsilon=0.5,
    )

    policy.epsilon = 0.2

    assert policy.epsilon == 0.2


def test_policy_decays_epsilon() -> None:
    policy = QTablePolicy(
        actions=[0, 1],
        epsilon=1.0,
    )

    result = policy.decay_epsilon(
        decay_rate=0.5,
        minimum_epsilon=0.1,
    )

    assert result == pytest.approx(0.5)
    assert policy.epsilon == pytest.approx(0.5)


def test_epsilon_does_not_fall_below_minimum() -> None:
    policy = QTablePolicy(
        actions=[0, 1],
        epsilon=0.2,
    )

    result = policy.decay_epsilon(
        decay_rate=0.1,
        minimum_epsilon=0.1,
    )

    assert result == pytest.approx(0.1)
    assert policy.epsilon == pytest.approx(0.1)


def test_q_table_uses_custom_state_encoder() -> None:
    def encoder(state: State) -> tuple[int, int]:
        return state.agent_position

    policy = QTablePolicy(
        actions=[0, 1],
        state_encoder=encoder,
    )

    first_state = make_state(
        x=1,
        y=2,
        has_key=False,
    )

    second_state = make_state(
        x=1,
        y=2,
        has_key=True,
    )

    policy.set_q_value(
        first_state,
        action=1,
        value=5.0,
    )

    assert policy.get_q_value(
        second_state,
        action=1,
    ) == 5.0

    assert policy.state_count == 1


def test_q_table_clear() -> None:
    policy = QTablePolicy(
        actions=[0, 1]
    )

    policy.q_values(
        make_state()
    )

    assert policy.state_count == 1

    policy.clear()

    assert policy.state_count == 0


def test_q_table_to_dict() -> None:
    policy = QTablePolicy(
        actions=[0, 1],
        learning_rate=0.2,
        discount_factor=0.9,
        epsilon=0.1,
    )

    state = make_state()

    policy.set_q_value(
        state,
        action=1,
        value=3.0,
    )

    data = policy.to_dict()

    assert data["actions"] == [0, 1]
    assert data["learning_rate"] == 0.2
    assert data["discount_factor"] == 0.9
    assert data["epsilon"] == 0.1
    assert data["state_count"] == 1
    assert len(data["q_table"]) == 1


def test_q_table_rejects_empty_actions() -> None:
    with pytest.raises(ValueError):
        QTablePolicy(
            actions=[]
        )


def test_q_table_rejects_duplicate_actions() -> None:
    with pytest.raises(ValueError):
        QTablePolicy(
            actions=[0, 1, 1]
        )


def test_q_table_rejects_non_integer_action() -> None:
    with pytest.raises(TypeError):
        QTablePolicy(
            actions=[0, "move"]  # type: ignore[list-item]
        )


def test_q_table_rejects_unknown_action() -> None:
    policy = QTablePolicy(
        actions=[0, 1]
    )

    with pytest.raises(ValueError):
        policy.get_q_value(
            make_state(),
            action=5,
        )


def test_q_table_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        QTablePolicy(
            actions=[0, 1],
            learning_rate=0.0,
        )

    with pytest.raises(ValueError):
        QTablePolicy(
            actions=[0, 1],
            discount_factor=1.5,
        )

    with pytest.raises(ValueError):
        QTablePolicy(
            actions=[0, 1],
            epsilon=-0.1,
        )


def test_q_table_rejects_invalid_update_values() -> None:
    policy = QTablePolicy(
        actions=[0, 1]
    )

    with pytest.raises(TypeError):
        policy.update(
            state=make_state(),
            action=0,
            reward="high",  # type: ignore[arg-type]
            next_state=make_state(x=2),
        )

    with pytest.raises(TypeError):
        policy.update(
            state=make_state(),
            action=0,
            reward=1.0,
            next_state=make_state(x=2),
            done="yes",  # type: ignore[arg-type]
        )


def test_q_table_rejects_unhashable_encoder_result() -> None:
    def invalid_encoder(state: State) -> list[int]:
        return [
            state.agent_position[0],
            state.agent_position[1],
        ]

    policy = QTablePolicy(
        actions=[0, 1],
        state_encoder=invalid_encoder,  # type: ignore[arg-type]
    )

    with pytest.raises(TypeError):
        policy.q_values(
            make_state()
        )