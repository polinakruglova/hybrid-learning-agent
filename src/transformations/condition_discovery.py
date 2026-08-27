from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from .effect_grouper import EffectGroup


@dataclass
class ConditionCandidate:
    """
    Кандидат на условие, которое отличает
    один результат действия от другого.

    Пример:

        feature = "x"
        value = 40

        probability_a = 0.05
        probability_b = 0.90

    Это означает:

        x == 40

    редко встречается в группе A,
    но часто встречается в группе B.
    """

    feature: str
    value: Any

    probability_a: float
    probability_b: float

    score: float

    observations_a: int
    observations_b: int

    def __str__(self) -> str:
        return (
            "ConditionCandidate("
            f"{self.feature} == {self.value!r}, "
            f"group_a={self.probability_a:.2f}, "
            f"group_b={self.probability_b:.2f}, "
            f"score={self.score:.2f}, "
            f"observations_a={self.observations_a}, "
            f"observations_b={self.observations_b}"
            ")"
        )


class ConditionDiscovery:
    """
    Ищет признаки состояния ДО действия,
    которые различают разные результаты
    одного и того же action.

    Схема:

        StateBefore
             +
           Action
             ↓
        EffectGroup A

        StateBefore
             +
        тот же Action
             ↓
        EffectGroup B

             ↓

        сравниваем StateBefore

             ↓

        ConditionCandidate
    """

    def compare_groups(
        self,
        group_a: EffectGroup,
        group_b: EffectGroup,
        min_support: float = 0.20,
        min_score: float = 0.30,
    ) -> list[ConditionCandidate]:
        """
        Сравнивает две группы результатов
        одного action.

        min_support:
            насколько часто значение должно
            встречаться хотя бы в одной группе.

        min_score:
            насколько сильно частота значения
            должна отличаться между группами.
        """

        # ------------------------------------------------------
        # Проверяем, что сравнивается один action
        # ------------------------------------------------------

        if group_a.action != group_b.action:
            raise ValueError(
                "Можно сравнивать только группы "
                "одного и того же action."
            )

        transitions_a = group_a.transitions
        transitions_b = group_b.transitions

        if not transitions_a or not transitions_b:
            return []

        # ------------------------------------------------------
        # Собираем все признаки StateBefore
        # ------------------------------------------------------

        feature_names: set[str] = set()

        for transition in transitions_a:
            feature_names.update(
                transition.before.keys()
            )

        for transition in transitions_b:
            feature_names.update(
                transition.before.keys()
            )

        candidates: list[ConditionCandidate] = []

        # ------------------------------------------------------
        # Сравниваем каждый признак
        # ------------------------------------------------------

        for feature in sorted(feature_names):

            values_a = self._collect_values(
                transitions=transitions_a,
                feature=feature,
            )

            values_b = self._collect_values(
                transitions=transitions_b,
                feature=feature,
            )

            if not values_a and not values_b:
                continue

            counter_a = Counter(values_a)
            counter_b = Counter(values_b)

            possible_values = (
                set(counter_a.keys())
                | set(counter_b.keys())
            )

            total_a = len(transitions_a)
            total_b = len(transitions_b)

            # --------------------------------------------------
            # Проверяем каждое возможное значение признака
            # --------------------------------------------------

            for value in possible_values:

                probability_a = (
                    counter_a[value] / total_a
                )

                probability_b = (
                    counter_b[value] / total_b
                )

                # Насколько часто значение вообще встречается
                # хотя бы в одной группе.
                support = max(
                    probability_a,
                    probability_b,
                )

                if support < min_support:
                    continue

                # Насколько сильно это значение
                # разделяет две группы.
                score = abs(
                    probability_a
                    - probability_b
                )

                if score < min_score:
                    continue

                candidate = ConditionCandidate(
                    feature=feature,
                    value=value,
                    probability_a=probability_a,
                    probability_b=probability_b,
                    score=score,
                    observations_a=total_a,
                    observations_b=total_b,
                )

                candidates.append(
                    candidate
                )

        # ------------------------------------------------------
        # Самые сильные условия ставим первыми
        # ------------------------------------------------------

        return sorted(
            candidates,
            key=lambda candidate: (
                candidate.score,
                max(
                    candidate.probability_a,
                    candidate.probability_b,
                ),
                candidate.observations_a
                + candidate.observations_b,
            ),
            reverse=True,
        )

    # ==========================================================
    # СБОР ЗНАЧЕНИЙ ОДНОГО ПРИЗНАКА
    # ==========================================================

    def _collect_values(
        self,
        transitions,
        feature: str,
    ) -> list[Any]:
        """
        Берёт значение указанного признака
        из StateBefore каждого перехода.
        """

        values: list[Any] = []

        for transition in transitions:

            if feature not in transition.before:
                continue

            raw_value = transition.before[
                feature
            ]

            value = self._make_comparable(
                raw_value
            )

            # Сложные объекты пока игнорируем.
            if value is None:
                continue

            values.append(
                value
            )

        return values

    # ==========================================================
    # ПРЕОБРАЗОВАНИЕ ЗНАЧЕНИЯ
    # ==========================================================

    @staticmethod
    def _make_comparable(
        value: Any,
    ) -> Any:
        """
        ConditionDiscovery первого уровня
        работает только с простыми признаками:

            int
            float
            str
            bool

        Карты, массивы и сложные структуры
        пока специально игнорируются.

        Позже для них сделаем отдельный
        пространственный анализ.
        """

        # Обычные Python-типы
        if isinstance(
            value,
            (
                str,
                int,
                float,
                bool,
            ),
        ):
            return value

        # numpy scalar -> Python scalar
        if hasattr(value, "item"):

            try:
                item = value.item()

            except (
                ValueError,
                TypeError,
            ):
                return None

            if isinstance(
                item,
                (
                    str,
                    int,
                    float,
                    bool,
                ),
            ):
                return item

        # ------------------------------------------------------
        # Списки, numpy arrays, карты и другие
        # сложные объекты пока не сравниваем.
        # ------------------------------------------------------

        return None