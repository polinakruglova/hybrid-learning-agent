from dataclasses import dataclass
from typing import Any

from src.state import State


@dataclass(frozen=True, slots=True)
class Experience:
    """
    Один переход агента в среде.

    state:
        Состояние до действия.

    action:
        Выполненное действие.

    reward:
        Полученная награда.

    next_state:
        Состояние после действия.

    done:
        Завершился ли эпизод после этого перехода.

    success:
        Был ли переход успешным с точки зрения задачи.

    source:
        Источник опыта:
        - agent;
        - teacher;
        - rule;
        - manual.
    """

    state: State
    action: int
    reward: float
    next_state: State
    done: bool = False
    success: bool = False
    source: str = "agent"

    def __post_init__(self) -> None:
        if not isinstance(self.action, int):
            raise TypeError("action должен быть целым числом.")

        if not isinstance(self.reward, (int, float)):
            raise TypeError("reward должен быть числом.")

        if not isinstance(self.done, bool):
            raise TypeError("done должен иметь тип bool.")

        if not isinstance(self.success, bool):
            raise TypeError("success должен иметь тип bool.")

        if not isinstance(self.source, str):
            raise TypeError("source должен быть строкой.")

        if not self.source.strip():
            raise ValueError("source не может быть пустой строкой.")

    @property
    def state_changed(self) -> bool:
        """
        Изменилось ли состояние после действия.
        """
        return self.state != self.next_state

    @property
    def is_positive(self) -> bool:
        """
        Получил ли агент положительную награду.
        """
        return self.reward > 0

    @property
    def is_negative(self) -> bool:
        """
        Получил ли агент отрицательную награду.
        """
        return self.reward < 0

    @property
    def is_neutral(self) -> bool:
        """
        Была ли награда нулевой.
        """
        return self.reward == 0

    def to_dict(self) -> dict[str, Any]:
        """
        Преобразует переход в сериализуемый словарь.
        """
        return {
            "state": self.state.to_dict(),
            "action": self.action,
            "reward": float(self.reward),
            "next_state": self.next_state.to_dict(),
            "done": self.done,
            "success": self.success,
            "source": self.source,
            "state_changed": self.state_changed,
        }