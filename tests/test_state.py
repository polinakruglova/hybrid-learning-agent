import pytest

from src.state import State


def test_state_creation() -> None:
    state = State(
        agent_position=(1, 3),
        agent_direction=1,
        key_position=(1, 2),
        door_position=(2, 2),
        goal_position=(3, 3),
        key_visible=True,
        door_visible=True,
        goal_visible=True,
        door_locked=True,
    )

    assert state.agent_x == 1
    assert state.agent_y == 3
    assert state.key_position == (1, 2)
    assert state.has_key is False
    assert state.door_locked is True


def test_state_is_hashable() -> None:
    state = State(
        agent_position=(1, 1),
        agent_direction=0,
    )

    q_table = {
        state: [0.0, 0.5, 0.0],
    }

    assert q_table[state][1] == 0.5


def test_state_rejects_invalid_direction() -> None:
    with pytest.raises(ValueError):
        State(
            agent_position=(1, 1),
            agent_direction=5,
        )


def test_state_to_dict() -> None:
    state = State(
        agent_position=(2, 4),
        agent_direction=3,
        has_key=True,
    )

    data = state.to_dict()

    assert data["agent_x"] == 2
    assert data["agent_y"] == 4
    assert data["has_key"] is True