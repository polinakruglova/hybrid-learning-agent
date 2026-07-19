from collections.abc import Iterable, Mapping
from typing import Any

from src.rules.rule import Rule


class RuleStore:
    """
    Хранилище правил агента.

    Отвечает за:
    - добавление правил;
    - поиск подходящих правил;
    - выбор лучшего правила;
    - удаление слабых правил.
    """

    def __init__(self, rules: Iterable[Rule] | None = None) -> None:
        self._rules: list[Rule] = list(rules or [])

    @property
    def rules(self) -> tuple[Rule, ...]:
        """
        Возвращает правила в неизменяемом виде.
        """
        return tuple(self._rules)

    def __len__(self) -> int:
        return len(self._rules)

    def add(self, rule: Rule) -> None:
        """
        Добавляет правило в хранилище.

        Полный дубликат не добавляется повторно.
        """
        if self.contains_equivalent(rule):
            return

        self._rules.append(rule)

    def extend(self, rules: Iterable[Rule]) -> None:
        """
        Добавляет несколько правил.
        """
        for rule in rules:
            self.add(rule)

    def contains_equivalent(self, candidate: Rule) -> bool:
        """
        Проверяет наличие эквивалентного правила.
        """
        return self.find_equivalent(candidate) is not None

    def find_equivalent(
        self,
        candidate: Rule,
    ) -> Rule | None:
        """
        Возвращает существующее правило с теми же
        условиями и действием.
        """
        for rule in self._rules:
            if (
                rule.conditions == candidate.conditions
                and rule.action == candidate.action
            ):
                return rule

        return None

    def find_matching(
        self,
        predicates: Mapping[str, Any],
    ) -> list[Rule]:
        """
        Возвращает все правила,
        условия которых выполняются.
        """
        return [
            rule
            for rule in self._rules
            if rule.matches(predicates)
        ]

    def select_best(
        self,
        predicates: Mapping[str, Any],
    ) -> Rule | None:
        """
        Выбирает лучшее подходящее правило.

        Приоритет:
        1. confidence;
        2. average_reward;
        3. specificity;
        4. support.
        """
        matching_rules = self.find_matching(predicates)

        if not matching_rules:
            return None

        return max(
            matching_rules,
            key=self._selection_score,
        )

    @staticmethod
    def _selection_score(rule: Rule) -> tuple[float, float, int, int]:
        return (
            rule.confidence,
            rule.average_reward,
            rule.specificity,
            rule.support,
        )

    def remove(self, rule: Rule) -> bool:
        """
        Удаляет правило.

        Возвращает True, если правило было найдено.
        """
        try:
            self._rules.remove(rule)
        except ValueError:
            return False

        return True

    def clear(self) -> None:
        """
        Удаляет все правила.
        """
        self._rules.clear()

    def prune(
        self,
        minimum_support: int = 10,
        minimum_confidence: float = 0.2,
        minimum_average_reward: float | None = None,
    ) -> list[Rule]:
        """
        Удаляет слабые правила.

        Правило удаляется только после накопления
        minimum_support наблюдений.

        Возвращает список удалённых правил.
        """
        if minimum_support < 0:
            raise ValueError(
                "minimum_support не может быть отрицательным."
            )

        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError(
                "minimum_confidence должен быть от 0 до 1."
            )

        removed: list[Rule] = []
        retained: list[Rule] = []

        for rule in self._rules:
            has_enough_support = rule.support >= minimum_support

            low_confidence = (
                rule.confidence < minimum_confidence
            )

            low_reward = (
                minimum_average_reward is not None
                and rule.average_reward < minimum_average_reward
            )

            should_remove = (
                has_enough_support
                and (low_confidence or low_reward)
            )

            if should_remove:
                removed.append(rule)
            else:
                retained.append(rule)

        self._rules = retained

        return removed

    def to_list(self) -> list[dict[str, Any]]:
        """
        Преобразует все правила в список словарей.
        """
        return [
            rule.to_dict()
            for rule in self._rules
        ]