import pytest

from src.state import StateParser


def test_parser_creates_state_from_old_format() -> None:
    raw_state = {
        "agent_x": 1,
        "agent_y": 3,
        "agent_dir": 1,
        "key_x": 1,
        "key_y": 2,
        "door_x": 2,
        "door_y": 2,
        "goal_x": 3,
        "goal_y": 3,
        "has_key": False,
        "key_visible": True,
        "door_visible": True,
        "goal_visible": True,
        "door_open": False,
        "door_locked": True,
    }

    state = StateParser.parse(raw_state)

    assert state.agent_position == (1, 3)
    assert state.agent_direction == 1
    assert state.key_position == (1, 2)
    assert state.door_position == (2, 2)
    assert state.goal_position == (3, 3)
    assert state.door_locked is True


def test_parser_accepts_agent_direction_name() -> None:
    raw_state = {
        "agent_x": 2,
        "agent_y": 4,
        "agent_direction": 3,
    }

    state = StateParser.parse(raw_state)

    assert state.agent_position == (2, 4)
    assert state.agent_direction == 3


def test_parser_converts_negative_position_to_none() -> None:
    raw_state = {
        "agent_x": 1,
        "agent_y": 1,
        "agent_dir": 0,
        "key_x": -1,
        "key_y": -1,
    }

    state = StateParser.parse(raw_state)

    assert state.key_position is None


def test_parser_rejects_missing_agent_coordinate() -> None:
    raw_state = {
        "agent_y": 1,
        "agent_dir": 0,
    }

    with pytest.raises(ValueError):
        StateParser.parse(raw_state)


def test_parser_rejects_incomplete_object_position() -> None:
    raw_state = {
        "agent_x": 1,
        "agent_y": 1,
        "agent_dir": 0,
        "door_x": 2,
    }

    with pytest.raises(ValueError):
        StateParser.parse(raw_state)