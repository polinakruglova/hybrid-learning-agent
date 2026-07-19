from src.predicates import PredicateGenerator
from src.state import State


def test_generator_creates_basic_state_predicates() -> None:
    state = State(
        agent_position=(1, 3),
        agent_direction=1,
        has_key=True,
        door_open=False,
        door_locked=True,
        key_visible=True,
        door_visible=True,
        goal_visible=True,
    )

    generator = PredicateGenerator()
    predicates = generator.generate(state)

    assert predicates["has_key"] is True
    assert predicates["door_open"] is False
    assert predicates["door_locked"] is True
    assert predicates["key_visible"] is True


def test_generator_calculates_key_position() -> None:
    state = State(
        agent_position=(1, 3),
        agent_direction=1,
        key_position=(1, 2),
    )

    generator = PredicateGenerator()
    predicates = generator.generate(state)

    assert predicates["distance_to_key"] == 1
    assert predicates["near_key"] is True
    assert predicates["very_close_to_key"] is True
    assert predicates["key_above"] is True
    assert predicates["key_below"] is False
    assert predicates["same_column_with_key"] is True


def test_generator_calculates_door_position() -> None:
    state = State(
        agent_position=(1, 3),
        agent_direction=1,
        door_position=(3, 3),
    )

    generator = PredicateGenerator()
    predicates = generator.generate(state)

    assert predicates["distance_to_door"] == 2
    assert predicates["near_door"] is False
    assert predicates["door_right"] is True
    assert predicates["same_row_with_door"] is True


def test_generator_detects_agent_at_goal() -> None:
    state = State(
        agent_position=(3, 3),
        agent_direction=0,
        goal_position=(3, 3),
    )

    generator = PredicateGenerator()
    predicates = generator.generate(state)

    assert predicates["distance_to_goal"] == 0
    assert predicates["at_goal"] is True
    assert predicates["near_goal"] is True
    assert predicates["very_close_to_goal"] is False


def test_generator_detects_far_object() -> None:
    state = State(
        agent_position=(0, 0),
        agent_direction=0,
        goal_position=(4, 4),
    )

    generator = PredicateGenerator()
    predicates = generator.generate(state)

    assert predicates["distance_to_goal"] == 8
    assert predicates["far_from_goal"] is True
    assert predicates["goal_right"] is True
    assert predicates["goal_below"] is True


def test_generator_handles_unknown_object_position() -> None:
    state = State(
        agent_position=(1, 1),
        agent_direction=0,
        key_position=None,
    )

    generator = PredicateGenerator()
    predicates = generator.generate(state)

    assert predicates["distance_to_key"] is None
    assert predicates["euclidean_distance_to_key"] is None
    assert predicates["near_key"] is False
    assert predicates["key_left"] is False
    assert predicates["same_row_with_key"] is False


def test_generator_does_not_modify_state() -> None:
    state = State(
        agent_position=(1, 3),
        agent_direction=1,
        key_position=(1, 2),
    )

    generator = PredicateGenerator()
    generator.generate(state)

    assert state.agent_position == (1, 3)
    assert state.key_position == (1, 2)