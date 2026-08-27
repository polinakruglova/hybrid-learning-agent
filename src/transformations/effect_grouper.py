from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from .transition_memory import Transition


@dataclass
class EffectGroup:
    """
    Группа переходов, в которых одно и то же действие
    привело к одинаковому результату.
    """

    action: Any
    effect_signature: tuple
    transitions: list[Transition]

    @property
    def count(self) -> int:
        return len(self.transitions)


class EffectGrouper:
    """
    Группирует опыт по схеме:

        ACTION
          ↓
    одинаковые эффекты
          ↓
      EffectGroup

    Пока мы НЕ ищем причину различий.
    Только обнаруживаем, что одно действие
    может иметь несколько результатов.
    """

    def group_by_action(
        self,
        transitions: list[Transition],
        action: Any,
    ) -> list[EffectGroup]:

        if hasattr(action, "item"):
            action = action.item()

        groups: dict[
            tuple,
            list[Transition],
        ] = defaultdict(list)

        for transition in transitions:

            if transition.action != action:
                continue

            signature = self._make_effect_signature(
                transition
            )

            groups[signature].append(
                transition
            )

        result = []

        for signature, group_transitions in groups.items():

            result.append(
                EffectGroup(
                    action=action,
                    effect_signature=signature,
                    transitions=group_transitions,
                )
            )

        return sorted(
            result,
            key=lambda group: group.count,
            reverse=True,
        )


    def group_all(
        self,
        transitions: list[Transition],
    ) -> dict[Any, list[EffectGroup]]:
        """
        Автоматически группирует результаты
        для всех обнаруженных действий.
        """

        actions = {
            transition.action
            for transition in transitions
        }

        return {
            action: self.group_by_action(
                transitions,
                action,
            )
            for action in sorted(actions)
        }


    @staticmethod
    def _make_effect_signature(
        transition: Transition,
    ) -> tuple:
        """
        Создаёт описание результата перехода.

        Например:

            (
                ("x", "delta", 1),
            )

        или:

            (
                ("message", "set", b"It's solid stone."),
            )

        Если ничего не изменилось:

            ("NO_CHANGE",)
        """

        effects = []

        for change in transition.changes:

            if change.delta is not None:

                effect = (
                    change.name,
                    "delta",
                    change.delta,
                )

            else:

                effect = (
                    change.name,
                    "set",
                    EffectGrouper._make_hashable(
                        change.after
                    ),
                )

            effects.append(effect)

        if not effects:
            return ("NO_CHANGE",)

        return tuple(
            sorted(
                effects,
                key=lambda item: item[0],
            )
        )


    @staticmethod
    def _make_hashable(
        value: Any,
    ) -> Any:
        """
        Некоторые значения среды могут быть
        списками, массивами и другими
        нехешируемыми объектами.

        Для сигнатуры превращаем их
        в безопасное представление.
        """

        try:
            hash(value)
            return value

        except TypeError:
            return repr(value)
