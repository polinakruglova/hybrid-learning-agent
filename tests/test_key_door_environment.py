import pytest

from src.environments import (
    KeyDoorEnvironment,
    StepResult,
)
from src.state import State


def test_environment_creation() -> None:
    environment = KeyDoorEnvironment(
        width=7,
        height=7,
        max_steps=50,
    )

    assert environment.width == 7
    assert environment.height == 7
    assert environment.max_steps == 50
    assert environment.step_count == 0
    assert environment.done is False
    assert environment.success is False
    assert environment.has_key is False
    assert environment.door_open is False
    assert environment.door_locked is True


def test_reset_returns_state() -> None:
    environment = KeyDoorEnvironment()

    state = environment.reset()

    assert isinstance(state, State)
    assert state.agent_position == (1, 1)
    assert state.agent_direction == (
        KeyDoorEnvironment.ACTION_RIGHT
    )
    assert state.has_key is False
    assert state.door_open is False
    assert state.door_locked is True
    assert state.key_position is not None


def test_get_state_contains_environment_data() -> None:
    environment = KeyDoorEnvironment()

    state = environment.get_state()

    assert state.agent_position == (
        environment.agent_position
    )
    assert state.key_position == (
        environment.key_position
    )
    assert state.door_position == (
        environment.door_position
    )
    assert state.goal_position == (
        environment.goal_position
    )


def test_move_right() -> None:
    environment = KeyDoorEnvironment()

    result = environment.step(
        KeyDoorEnvironment.ACTION_RIGHT
    )

    assert isinstance(result, StepResult)
    assert result.state.agent_position == (2, 1)
    assert result.reward == pytest.approx(-0.01)
    assert result.info["event"] == "moved"


def test_wall_blocks_movement() -> None:
    environment = KeyDoorEnvironment()

    result = environment.step(
        KeyDoorEnvironment.ACTION_UP
    )

    assert result.state.agent_position == (1, 1)
    assert result.reward == pytest.approx(-0.10)
    assert result.info["event"] == "wall_collision"


def test_closed_door_blocks_movement() -> None:
    environment = KeyDoorEnvironment()

    environment._agent_position = (
        environment.door_position[0] - 1,
        environment.door_position[1],
    )

    result = environment.step(
        KeyDoorEnvironment.ACTION_RIGHT
    )

    assert result.state.agent_position == (
        environment.door_position[0] - 1,
        environment.door_position[1],
    )

    assert result.reward == pytest.approx(-0.20)
    assert (
        result.info["event"]
        == "closed_door_collision"
    )


def test_cannot_pickup_distant_key() -> None:
    environment = KeyDoorEnvironment()

    result = environment.step(
        KeyDoorEnvironment.ACTION_PICKUP
    )

    assert result.reward == pytest.approx(-0.05)
    assert result.state.has_key is False
    assert result.info["event"] == "key_not_near"


def test_pickup_nearby_key() -> None:
    environment = KeyDoorEnvironment()

    environment._agent_position = (
        environment.key_position[0],
        environment.key_position[1] - 1,
    )

    result = environment.step(
        KeyDoorEnvironment.ACTION_PICKUP
    )

    assert result.reward == pytest.approx(1.0)
    assert result.state.has_key is True
    assert result.state.key_position is None
    assert result.state.key_visible is False
    assert result.info["event"] == "key_picked_up"


def test_cannot_open_door_without_key() -> None:
    environment = KeyDoorEnvironment()

    environment._agent_position = (
        environment.door_position[0] - 1,
        environment.door_position[1],
    )

    result = environment.step(
        KeyDoorEnvironment.ACTION_OPEN_DOOR
    )

    assert result.reward == pytest.approx(-0.20)
    assert result.state.door_open is False
    assert result.state.door_locked is True
    assert result.info["event"] == "door_locked"


def test_cannot_open_distant_door() -> None:
    environment = KeyDoorEnvironment()

    environment._has_key = True
    environment._key_position = None

    result = environment.step(
        KeyDoorEnvironment.ACTION_OPEN_DOOR
    )

    assert result.reward == pytest.approx(-0.05)
    assert result.state.door_open is False
    assert result.info["event"] == "door_not_near"


def test_open_door_with_key() -> None:
    environment = KeyDoorEnvironment()

    environment._has_key = True
    environment._key_position = None

    environment._agent_position = (
        environment.door_position[0] - 1,
        environment.door_position[1],
    )

    result = environment.step(
        KeyDoorEnvironment.ACTION_OPEN_DOOR
    )

    assert result.reward == pytest.approx(2.0)
    assert result.state.door_open is True
    assert result.state.door_locked is False
    assert result.info["event"] == "door_opened"


def test_move_through_open_door() -> None:
    environment = KeyDoorEnvironment()

    environment._door_open = True

    environment._agent_position = (
        environment.door_position[0] - 1,
        environment.door_position[1],
    )

    result = environment.step(
        KeyDoorEnvironment.ACTION_RIGHT
    )

    assert (
        result.state.agent_position
        == environment.door_position
    )
    assert result.info["event"] == "moved"


def test_reach_goal() -> None:
    environment = KeyDoorEnvironment()

    environment._door_open = True

    environment._agent_position = (
        environment.goal_position[0] - 1,
        environment.goal_position[1],
    )

    result = environment.step(
        KeyDoorEnvironment.ACTION_RIGHT
    )

    assert result.done is True
    assert result.success is True
    assert result.reward == pytest.approx(10.0)
    assert result.info["event"] == "goal_reached"


def test_max_steps_ends_episode() -> None:
    environment = KeyDoorEnvironment(
        max_steps=2
    )

    environment.step(
        KeyDoorEnvironment.ACTION_WAIT
    )

    result = environment.step(
        KeyDoorEnvironment.ACTION_WAIT
    )

    assert result.done is True
    assert result.success is False
    assert result.reward == pytest.approx(-1.02)
    assert result.info["event"] == "max_steps_reached"


def test_step_after_done_raises_error() -> None:
    environment = KeyDoorEnvironment(
        max_steps=1
    )

    environment.step(
        KeyDoorEnvironment.ACTION_WAIT
    )

    with pytest.raises(RuntimeError):
        environment.step(
            KeyDoorEnvironment.ACTION_WAIT
        )


def test_reset_restores_environment() -> None:
    environment = KeyDoorEnvironment()

    environment._agent_position = (2, 2)
    environment._has_key = True
    environment._key_position = None
    environment._door_open = True
    environment._done = True
    environment._success = True
    environment._step_count = 50

    state = environment.reset()

    assert state.agent_position == (1, 1)
    assert state.has_key is False
    assert state.key_position is not None
    assert state.door_open is False
    assert state.door_locked is True

    assert environment.step_count == 0
    assert environment.done is False
    assert environment.success is False


def test_render_returns_map() -> None:
    environment = KeyDoorEnvironment()

    rendered = environment.render()

    assert isinstance(rendered, str)
    assert "A" in rendered
    assert "K" in rendered
    assert "D" in rendered
    assert "G" in rendered
    assert "#" in rendered


def test_invalid_action_raises_value_error() -> None:
    environment = KeyDoorEnvironment()

    with pytest.raises(ValueError):
        environment.step(99)


def test_non_integer_action_raises_type_error() -> None:
    environment = KeyDoorEnvironment()

    with pytest.raises(TypeError):
        environment.step(
            "right"  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("width", "height", "max_steps"),
    [
        (4, 7, 100),
        (7, 4, 100),
        (7, 7, 0),
        (7, 7, -1),
    ],
)
def test_invalid_environment_configuration(
    width: int,
    height: int,
    max_steps: int,
) -> None:
    with pytest.raises(ValueError):
        KeyDoorEnvironment(
            width=width,
            height=height,
            max_steps=max_steps,
        )