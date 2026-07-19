import pytest

from src.memory import Experience, ExperienceBuffer
from src.state import State


def make_state(
    x: int = 1,
    y: int = 1,
    direction: int = 0,
) -> State:
    return State(
        agent_position=(x, y),
        agent_direction=direction,
    )


def make_experience(
    index: int,
    reward: float = 0.0,
    success: bool = False,
    source: str = "agent",
) -> Experience:
    return Experience(
        state=make_state(x=index),
        action=index,
        reward=reward,
        next_state=make_state(x=index + 1),
        success=success,
        source=source,
    )


def test_buffer_starts_empty() -> None:
    buffer = ExperienceBuffer(capacity=5)

    assert len(buffer) == 0
    assert buffer.capacity == 5
    assert buffer.experiences == ()


def test_buffer_adds_experience() -> None:
    buffer = ExperienceBuffer(capacity=5)
    experience = make_experience(1)

    buffer.add(experience)

    assert len(buffer) == 1
    assert buffer.experiences[0] is experience


def test_buffer_rejects_invalid_object() -> None:
    buffer = ExperienceBuffer()

    with pytest.raises(TypeError):
        buffer.add("invalid")  # type: ignore[arg-type]


def test_buffer_extends_with_multiple_experiences() -> None:
    buffer = ExperienceBuffer(capacity=5)

    experiences = [
        make_experience(1),
        make_experience(2),
        make_experience(3),
    ]

    buffer.extend(experiences)

    assert len(buffer) == 3
    assert buffer.experiences == tuple(experiences)


def test_buffer_removes_oldest_when_full() -> None:
    buffer = ExperienceBuffer(capacity=3)

    first = make_experience(1)
    second = make_experience(2)
    third = make_experience(3)
    fourth = make_experience(4)

    buffer.extend(
        [
            first,
            second,
            third,
            fourth,
        ]
    )

    assert len(buffer) == 3
    assert first not in buffer.experiences
    assert buffer.experiences == (
        second,
        third,
        fourth,
    )


def test_buffer_samples_experiences() -> None:
    buffer = ExperienceBuffer(
        capacity=10,
        seed=42,
    )

    experiences = [
        make_experience(index)
        for index in range(5)
    ]

    buffer.extend(experiences)

    sample = buffer.sample(3)

    assert len(sample) == 3
    assert len(set(sample)) == 3

    for experience in sample:
        assert experience in experiences


def test_buffer_rejects_invalid_sample_size() -> None:
    buffer = ExperienceBuffer(capacity=5)
    buffer.add(make_experience(1))

    with pytest.raises(ValueError):
        buffer.sample(0)

    with pytest.raises(ValueError):
        buffer.sample(2)


def test_buffer_returns_successful_experiences() -> None:
    successful = make_experience(
        1,
        success=True,
    )

    unsuccessful = make_experience(
        2,
        success=False,
    )

    buffer = ExperienceBuffer()
    buffer.extend(
        [
            successful,
            unsuccessful,
        ]
    )

    assert buffer.successful() == [successful]


def test_buffer_filters_positive_and_negative_rewards() -> None:
    positive = make_experience(
        1,
        reward=1.0,
    )

    negative = make_experience(
        2,
        reward=-0.5,
    )

    neutral = make_experience(
        3,
        reward=0.0,
    )

    buffer = ExperienceBuffer()
    buffer.extend(
        [
            positive,
            negative,
            neutral,
        ]
    )

    assert buffer.positive() == [positive]
    assert buffer.negative() == [negative]


def test_buffer_filters_by_source() -> None:
    teacher = make_experience(
        1,
        source="teacher",
    )

    agent = make_experience(
        2,
        source="agent",
    )

    buffer = ExperienceBuffer()
    buffer.extend(
        [
            teacher,
            agent,
        ]
    )

    assert buffer.by_source("teacher") == [teacher]
    assert buffer.by_source("agent") == [agent]


def test_buffer_rejects_invalid_source_filter() -> None:
    buffer = ExperienceBuffer()

    with pytest.raises(TypeError):
        buffer.by_source(123)  # type: ignore[arg-type]

    with pytest.raises(ValueError):
        buffer.by_source("   ")


def test_buffer_calculates_reward_statistics() -> None:
    buffer = ExperienceBuffer()

    buffer.extend(
        [
            make_experience(1, reward=1.0),
            make_experience(2, reward=0.5),
            make_experience(3, reward=-0.5),
        ]
    )

    assert buffer.total_reward == pytest.approx(1.0)
    assert buffer.average_reward == pytest.approx(1 / 3)


def test_buffer_calculates_success_rate() -> None:
    buffer = ExperienceBuffer()

    buffer.extend(
        [
            make_experience(1, success=True),
            make_experience(2, success=True),
            make_experience(3, success=False),
            make_experience(4, success=False),
        ]
    )

    assert buffer.success_rate == pytest.approx(0.5)


def test_empty_buffer_statistics_are_zero() -> None:
    buffer = ExperienceBuffer()

    assert buffer.total_reward == 0.0
    assert buffer.average_reward == 0.0
    assert buffer.success_rate == 0.0


def test_buffer_returns_statistics() -> None:
    buffer = ExperienceBuffer(capacity=10)

    buffer.extend(
        [
            make_experience(
                1,
                reward=1.0,
                success=True,
            ),
            make_experience(
                2,
                reward=-0.5,
                success=False,
            ),
        ]
    )

    statistics = buffer.statistics()

    assert statistics["size"] == 2
    assert statistics["capacity"] == 10
    assert statistics["total_reward"] == pytest.approx(0.5)
    assert statistics["average_reward"] == pytest.approx(0.25)
    assert statistics["success_rate"] == pytest.approx(0.5)
    assert statistics["successful_count"] == 1
    assert statistics["positive_count"] == 1
    assert statistics["negative_count"] == 1


def test_buffer_clear() -> None:
    buffer = ExperienceBuffer()
    buffer.add(make_experience(1))

    buffer.clear()

    assert len(buffer) == 0


def test_buffer_converts_to_list() -> None:
    buffer = ExperienceBuffer()

    buffer.add(
        make_experience(
            1,
            reward=0.5,
            success=True,
            source="teacher",
        )
    )

    data = buffer.to_list()

    assert len(data) == 1
    assert data[0]["action"] == 1
    assert data[0]["reward"] == 0.5
    assert data[0]["success"] is True
    assert data[0]["source"] == "teacher"


def test_buffer_rejects_invalid_capacity() -> None:
    with pytest.raises(ValueError):
        ExperienceBuffer(capacity=0)