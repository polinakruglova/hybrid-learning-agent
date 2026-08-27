from __future__ import annotations

import numpy as np

from src.state import (
    UniversalObservationParser,
    WorldState,
)


def test_parser_returns_world_state() -> None:
    parser = UniversalObservationParser(
        source_name="test_environment"
    )

    state = parser.parse(
        {
            "health": 10,
            "alive": True,
        },
        step=5,
    )

    assert isinstance(state, WorldState)
    assert state.source == "test_environment"
    assert state.step == 5

    assert state["observation.health"] == 10
    assert state["observation.alive"] is True


def test_parser_flattens_nested_mapping() -> None:
    parser = UniversalObservationParser()

    state = parser.parse(
        {
            "agent": {
                "position": {
                    "x": 2,
                    "y": 7,
                }
            }
        }
    )

    assert state[
        "observation.agent.position.x"
    ] == 2

    assert state[
        "observation.agent.position.y"
    ] == 7


def test_parser_handles_numpy_array() -> None:
    parser = UniversalObservationParser(
        max_array_items=4
    )

    state = parser.parse(
        {
            "values": np.array(
                [1, 2, 3, 4, 5],
                dtype=np.int32,
            )
        }
    )

    assert state[
        "observation.values.size"
    ] == 5

    assert state[
        "observation.values.minimum"
    ] == 1.0

    assert state[
        "observation.values.maximum"
    ] == 5.0

    assert state[
        "observation.values.value.0"
    ] == 1

    assert state[
        "observation.values.truncated"
    ] is True


def test_world_state_copy_is_independent() -> None:
    original = WorldState(
        attributes={"health": 10}
    )

    copied = original.copy()
    copied["health"] = 5

    assert original["health"] == 10
    assert copied["health"] == 5