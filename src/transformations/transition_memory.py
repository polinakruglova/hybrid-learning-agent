from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .change_detector import FeatureChange


@dataclass
class Transition:
    """
    Один наблюдавшийся переход мира:

        состояние ДО
            +
          действие
            ↓
        состояние ПОСЛЕ
    """

    before: dict[str, Any]
    action: Any
    after: dict[str, Any]
    changes: list[FeatureChange]


class TransitionMemory:
    """
    Хранилище полного опыта агента.

    В отличие от TransformationTracker,
    здесь мы не сворачиваем опыт в счётчики.

    Каждый переход сохраняется отдельно,
    чтобы позже можно было искать условия,
    при которых одно действие даёт
    разные результаты.
    """

    def __init__(self):

        self.transitions: list[Transition] = []


    def add(
        self,
        before: dict[str, Any],
        action: Any,
        after: dict[str, Any],
        changes: list[FeatureChange],
    ) -> None:

        if hasattr(action, "item"):
            action = action.item()

        transition = Transition(
            before=dict(before),
            action=action,
            after=dict(after),
            changes=list(changes),
        )

        self.transitions.append(
            transition
        )


    def get_by_action(
        self,
        action: Any,
    ) -> list[Transition]:
        """
        Возвращает ВСЕ случаи,
        когда агент выполнял указанное действие.
        """

        if hasattr(action, "item"):
            action = action.item()

        return [
            transition
            for transition in self.transitions
            if transition.action == action
        ]


    def __len__(self) -> int:

        return len(self.transitions)
