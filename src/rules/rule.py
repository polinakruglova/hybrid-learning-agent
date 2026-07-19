from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(slots=True)
class Rule:
    """
    Логическое правило поведения агента.

    conditions:
        Условия, при которых правило применимо.

    action:
        Действие, рекомендуемое правилом.

    support:
        Количество наблюдений, подтверждающих правило.

    successes:
        Количество успешных применений.

    total_reward:
        Сумма наград, полученных после применения правила.
    """

    conditions: dict[str, Any]
    action: int

    support: int = 0
    successes: int = 0
    total_reward: float = 0.0

    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.conditions:
            raise ValueError(
                "Правило должно содержать хотя бы одно условие."
            )

        if not isinstance(self.action, int):
            raise TypeError("action должен быть целым числом.")

        if self.support < 0:
            raise ValueError("support не может быть отрицательным.")

        if self.successes < 0:
            raise ValueError("successes не может быть отрицательным.")

        if self.successes > self.support:
            raise ValueError(
                "successes не может превышать support."
            )

    def matches(self, predicates: Mapping[str, Any]) -> bool:
        """
        Проверяет, выполняются ли все условия правила.
        """
        return all(
            predicates.get(name) == expected_value
            for name, expected_value in self.conditions.items()
        )

    def update(
        self,
        reward: float,
        success: bool,
    ) -> None:
        """
        Обновляет статистику после применения правила.
        """
        self.support += 1
        self.total_reward += float(reward)

        if success:
            self.successes += 1

    @property
    def confidence(self) -> float:
        """
        Доля успешных применений правила.
        """
        if self.support == 0:
            return 0.0

        return self.successes / self.support

    @property
    def average_reward(self) -> float:
        """
        Средняя награда за применение правила.
        """
        if self.support == 0:
            return 0.0

        return self.total_reward / self.support

    @property
    def specificity(self) -> int:
        """
        Количество условий в правиле.

        Чем условий больше, тем правило более конкретное.
        """
        return len(self.conditions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conditions": dict(self.conditions),
            "action": self.action,
            "support": self.support,
            "successes": self.successes,
            "total_reward": self.total_reward,
            "confidence": self.confidence,
            "average_reward": self.average_reward,
            "specificity": self.specificity,
            "metadata": dict(self.metadata),
        }