from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from random import Random
from typing import Any

from src.memory.experience import Experience


class ExperienceBuffer:
    """
    Буфер опыта агента.

    Хранит переходы вида:
    state -> action -> reward -> next_state

    При переполнении удаляет самые старые записи.
    """

    def __init__(
        self,
        capacity: int = 10_000,
        seed: int | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError(
                "capacity должен быть положительным числом."
            )

        self._capacity = capacity
        self._buffer: deque[Experience] = deque(
            maxlen=capacity
        )
        self._random = Random(seed)

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def experiences(self) -> tuple[Experience, ...]:
        """
        Возвращает опыт в неизменяемом виде.
        """
        return tuple(self._buffer)

    def __len__(self) -> int:
        return len(self._buffer)

    def __iter__(self):
        return iter(self._buffer)

    def add(self, experience: Experience) -> None:
        """
        Добавляет один переход.
        """
        if not isinstance(experience, Experience):
            raise TypeError(
                "В буфер можно добавлять только Experience."
            )

        self._buffer.append(experience)

    def extend(
        self,
        experiences: Iterable[Experience],
    ) -> None:
        """
        Добавляет несколько переходов.
        """
        for experience in experiences:
            self.add(experience)

    def clear(self) -> None:
        """
        Полностью очищает память.
        """
        self._buffer.clear()

    def sample(
        self,
        batch_size: int,
    ) -> list[Experience]:
        """
        Возвращает случайную выборку без повторений.
        """
        if batch_size <= 0:
            raise ValueError(
                "batch_size должен быть положительным."
            )

        if batch_size > len(self._buffer):
            raise ValueError(
                "batch_size не может превышать размер буфера."
            )

        return self._random.sample(
            list(self._buffer),
            batch_size,
        )

    def successful(self) -> list[Experience]:
        """
        Возвращает переходы, помеченные как успешные.
        """
        return [
            experience
            for experience in self._buffer
            if experience.success
        ]

    def positive(self) -> list[Experience]:
        """
        Возвращает переходы с положительной наградой.
        """
        return [
            experience
            for experience in self._buffer
            if experience.is_positive
        ]

    def negative(self) -> list[Experience]:
        """
        Возвращает переходы с отрицательной наградой.
        """
        return [
            experience
            for experience in self._buffer
            if experience.is_negative
        ]

    def by_source(
        self,
        source: str,
    ) -> list[Experience]:
        """
        Возвращает опыт указанного источника.
        """
        if not isinstance(source, str):
            raise TypeError("source должен быть строкой.")

        normalized_source = source.strip()

        if not normalized_source:
            raise ValueError(
                "source не может быть пустой строкой."
            )

        return [
            experience
            for experience in self._buffer
            if experience.source == normalized_source
        ]

    @property
    def total_reward(self) -> float:
        """
        Сумма всех наград в памяти.
        """
        return sum(
            experience.reward
            for experience in self._buffer
        )

    @property
    def average_reward(self) -> float:
        """
        Средняя награда по всем переходам.
        """
        if not self._buffer:
            return 0.0

        return self.total_reward / len(self._buffer)

    @property
    def success_rate(self) -> float:
        """
        Доля успешных переходов.
        """
        if not self._buffer:
            return 0.0

        successful_count = sum(
            experience.success
            for experience in self._buffer
        )

        return successful_count / len(self._buffer)

    def statistics(self) -> dict[str, Any]:
        """
        Возвращает основную статистику памяти.
        """
        return {
            "size": len(self._buffer),
            "capacity": self.capacity,
            "total_reward": self.total_reward,
            "average_reward": self.average_reward,
            "success_rate": self.success_rate,
            "successful_count": len(self.successful()),
            "positive_count": len(self.positive()),
            "negative_count": len(self.negative()),
        }

    def to_list(self) -> list[dict[str, Any]]:
        """
        Преобразует весь опыт в список словарей.
        """
        return [
            experience.to_dict()
            for experience in self._buffer
        ]