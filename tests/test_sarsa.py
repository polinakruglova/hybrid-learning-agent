from __future__ import annotations

import pytest

from src.environments import KeyDoorEnvironment
from src.rl import SARSAPolicy


def create_environment() -> KeyDoorEnvironment:
    return KeyDoorEnvironment(
        width=7,
        height=7,
        max_steps=100,
    )


def create_policy(
    epsilon: float = 0.0,
) -> SARSAPolicy:
    environment = create_environment()

    return SARSAPolicy(
        actions=environment.ACTIONS,
        learning_rate=0.5,
        discount_factor=0.9,
        epsilon=epsilon,
        seed=42,
    )


def test_policy_contains_all_actions() -> None:
    environment = create_environment()
    policy = create_policy()

    assert policy.actions == environment.ACTIONS


def test_new_state_has_zero_q_values() -> None:
    environment = create_environment()
    state = environment.reset()

    policy = create_policy()

    values = policy.q_values(state)

    assert set(values) == set(
        environment.ACTIONS
    )

    assert all(
        value == 0.0
        for value in values.values()
    )


def test_terminal_update_uses_reward_only() -> None:
    environment = create_environment()
    state = environment.reset()

    policy = create_policy()

    updated_value = policy.update(
        state=state,
        action=environment.ACTION_WAIT,
        reward=1.0,
        next_state=state,
        next_action=None,
        done=True,
    )

    assert updated_value == pytest.approx(0.5)


def test_sarsa_update_uses_selected_next_action() -> None:
    environment = create_environment()

    state = environment.reset()

    step_result = environment.step(
        environment.ACTION_WAIT
    )

    next_state = step_result.state

    policy = create_policy()

    policy.set_q_value(
        next_state,
        environment.ACTION_RIGHT,
        2.0,
    )

    updated_value = policy.update(
        state=state,
        action=environment.ACTION_WAIT,
        reward=1.0,
        next_state=next_state,
        next_action=environment.ACTION_RIGHT,
        done=False,
    )

    expected_target = 1.0 + 0.9 * 2.0
    expected_value = 0.5 * expected_target

    assert updated_value == pytest.approx(
        expected_value
    )


def test_update_requires_next_action() -> None:
    environment = create_environment()
    state = environment.reset()

    policy = create_policy()

    with pytest.raises(
        ValueError,
        match="next_action",
    ):
        policy.update(
            state=state,
            action=environment.ACTION_WAIT,
            reward=0.0,
            next_state=state,
            next_action=None,
            done=False,
        )


def test_best_action_uses_highest_q_value() -> None:
    environment = create_environment()
    state = environment.reset()

    policy = create_policy()

    policy.set_q_value(
        state,
        environment.ACTION_LEFT,
        5.0,
    )

    assert (
        policy.best_action(state)
        == environment.ACTION_LEFT
    )


def test_choose_action_without_exploration() -> None:
    environment = create_environment()
    state = environment.reset()

    policy = create_policy(
        epsilon=1.0
    )

    policy.set_q_value(
        state,
        environment.ACTION_PICKUP,
        10.0,
    )

    action = policy.choose_action(
        state,
        explore=False,
    )

    assert action == environment.ACTION_PICKUP


def test_epsilon_decay_respects_minimum() -> None:
    policy = create_policy(
        epsilon=0.5
    )

    for _ in range(100):
        policy.decay_epsilon(
            decay_rate=0.5,
            minimum_epsilon=0.05,
        )

    assert policy.epsilon == pytest.approx(
        0.05
    )


def test_clear_removes_states() -> None:
    environment = create_environment()
    state = environment.reset()

    policy = create_policy()

    policy.ensure_state(state)

    assert policy.state_count == 1

    policy.clear()

    assert policy.state_count == 0


def test_unknown_action_is_rejected() -> None:
    environment = create_environment()
    state = environment.reset()

    policy = create_policy()

    with pytest.raises(
        ValueError,
        match="unknown action",
    ):
        policy.get_q_value(
            state,
            action=999,
        )