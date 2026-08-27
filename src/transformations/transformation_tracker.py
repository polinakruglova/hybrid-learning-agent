from __future__ import annotations

from collections import defaultdict
from typing import Any

from .change_detector import FeatureChange
from .transformation_rule import TransformationRule


class TransformationTracker:
    """
    Наблюдает изменения мира после действий.

    Сначала накапливает повторяющиеся эффекты,
    затем превращает устойчивые закономерности
    в TransformationRule.
    """

    def __init__(self):

        # Сколько вообще раз наблюдали каждое действие
        self.action_counts: dict[Any, int] = defaultdict(int)

        # Эффекты каждого действия
        #
        # Ключ:
        # (
        #     action,
        #     feature,
        #     change_type,
        #     value,
        # )
        #
        # Например:
        # (1, "x", "delta", 1)
        self.effect_counts: dict[tuple, int] = defaultdict(int)


    def observe(
        self,
        action: Any,
        changes: list[FeatureChange],
    ) -> None:
        """
        Добавляет один переход мира в статистику.
        """

        # numpy.int64 превращаем в обычный int,
        # чтобы правила были чистыми Python-объектами
        if hasattr(action, "item"):
            action = action.item()

        self.action_counts[action] += 1

        for change in changes:

            if change.delta is not None:

                signature = (
                    action,
                    change.name,
                    "delta",
                    change.delta,
                )

            else:

                signature = (
                    action,
                    change.name,
                    "set",
                    change.after,
                )

            self.effect_counts[signature] += 1


    def get_repeated(
        self,
        min_count: int = 3,
    ) -> list[tuple]:
        """
        Старый интерфейс оставляем,
        чтобы существующий скрипт продолжал работать.
        """

        result = []

        for signature, count in self.effect_counts.items():

            if count >= min_count:

                result.append(
                    (
                        signature,
                        count,
                    )
                )

        return sorted(
            result,
            key=lambda item: item[1],
            reverse=True,
        )


    def build_rules(
        self,
        min_count: int = 3,
        min_confidence: float = 0.5,
    ) -> list[TransformationRule]:
        """
        Превращает повторяющиеся эффекты
        в TransformationRule.
        """

        rules = []

        for signature, count in self.effect_counts.items():

            if count < min_count:
                continue

            (
                action,
                feature,
                change_type,
                value,
            ) = signature

            total_action_count = self.action_counts[action]

            if total_action_count == 0:
                continue

            confidence = count / total_action_count

            if confidence < min_confidence:
                continue

            rule = TransformationRule(
                action=action,

                conditions={},

                effects={
                    feature: (
                        change_type,
                        value,
                    )
                },

                observations=count,

                confidence=confidence,
            )

            rules.append(rule)


        return sorted(
            rules,
            key=lambda rule: (
                rule.confidence,
                rule.observations,
            ),
            reverse=True,
        )