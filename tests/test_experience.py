import pytest

from src.memory import Experience
from src.state import State


def make_state(
    x: int = 1,
    y: int = 1,
    direction: int = 0,
    has_key: bool = False,
) -> State:
    return State(
        agent_position=(x, y),
        agent_direction=direction,
        has_key=has_key,
    )


def test_experience_creation() -> None:
    state = make_state()
    next_state = make_state(x=2)

    experience = Experience(
        state=state,
        action=2,
        reward=0.5,
        next_state=next_state,
    )

    assert experience.state is state
    assert experience.next_state is next_state
    assert experience.action == 2
    assert experience.reward == 0.5
    assert experience.done is False
    assert experience.success is False
    assert experience.source == "agent"


def test_experience_detects_state_change() -> None:
    experience = Experience(
        state=make_state(x=1),
        action=2,
        reward=0.0,
        next_state=make_state(x=2),
    )

    assert experience.state_changed is True


def test_experience_detects_unchanged_state() -> None:
    state = make_state()

    experience = Experience(
        state=state,
        action=0,
        reward=0.0,
        next_state=state,
    )

    assert experience.state_changed is False


def test_experience_classifies_positive_reward() -> None:
    experience = Experience(
        state=make_state(),
        action=2,
        reward=1.0,
        next_state=make_state(x=2),
    )

    assert experience.is_positive is True
    assert experience.is_negative is False
    assert experience.is_neutral is False


def test_experience_classifies_negative_reward() -> None:
    experience = Experience(
        state=make_state(),
        action=2,
        reward=-0.5,
        next_state=make_state(),
    )

    assert experience.is_positive is False
    assert experience.is_negative is True
    assert experience.is_neutral is False


def test_experience_classifies_neutral_reward() -> None:
    experience = Experience(
        state=make_state(),
        action=0,
        reward=0.0,
        next_state=make_state(),
    )

    assert experience.is_positive is False
    assert experience.is_negative is False
    assert experience.is_neutral is True


def test_experience_accepts_teacher_source() -> None:
    experience = Experience(
        state=make_state(),
        action=3,
        reward=1.0,
        next_state=make_state(has_key=True),
        success=True,
        source="teacher",
    )

    assert experience.source == "teacher"
    assert experience.success is True


def test_experience_rejects_non_integer_action() -> None:
    with pytest.raises(TypeError):
        Experience(
            state=make_state(),
            action="move",  # type: ignore[arg-type]
            reward=0.0,
            next_state=make_state(),
        )


def test_experience_rejects_invalid_reward() -> None:
    with pytest.raises(TypeError):
        Experience(
            state=make_state(),
            action=2,
            reward="high",  # type: ignore[arg-type]
            next_state=make_state(),
        )


def test_experience_rejects_empty_source() -> None:
    with pytest.raises(ValueError):
        Experience(
            state=make_state(),
            action=2,
            reward=0.0,
            next_state=make_state(),
            source="   ",
        )


def test_experience_to_dict() -> None:
    experience = Experience(
        state=make_state(x=1, y=2),
        action=2,
        reward=0.75,
        next_state=make_state(x=2, y=2),
        done=True,
        success=True,
        source="teacher",
    )

    data = experience.to_dict()

    assert data["state"]["agent_x"] == 1
    assert data["next_state"]["agent_x"] == 2
    assert data["action"] == 2
    assert data["reward"] == 0.75
    assert data["done"] is True
    assert data["success"] is True
    assert data["source"] == "teacher"
    assert data["state_changed"] is True