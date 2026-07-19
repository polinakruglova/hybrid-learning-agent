from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.predicates import PredicateGenerator
from src.rules import Rule, RuleStore
from src.state import State


FallbackPolicy = Callable[[State], int | None]


class RuleAgent:
    """
    Агент, выбирающий действия на основе символических правил.

    Алгоритм:
    1. Получает State.
    2. Создаёт предикаты через PredicateGenerator.
    3. Ищет подходящие правила в RuleStore.
    4. Выбирает лучшее правило.
    5. Возвращает действие правила.

    Если подходящего правила нет, может:
    - вернуть fallback_action;
    - вызвать fallback_policy;
    - вернуть None.
    """

    def __init__(
        self,
        rule_store: RuleStore,
        predicate_generator: PredicateGenerator | None = None,
        fallback_action: int | None = None,
        fallback_policy: FallbackPolicy | None = None,
    ) -> None:
        if not isinstance(rule_store, RuleStore):
            raise TypeError(
                "rule_store должен быть объектом RuleStore."
            )

        if (
            fallback_action is not None
            and not isinstance(fallback_action, int)
        ):
            raise TypeError(
                "fallback_action должен быть int или None."
            )

        if (
            fallback_policy is not None
            and not callable(fallback_policy)
        ):
            raise TypeError(
                "fallback_policy должен быть вызываемым объектом."
            )

        self._rule_store = rule_store

        self._predicate_generator = (
            predicate_generator or PredicateGenerator()
        )

        self._fallback_action = fallback_action
        self._fallback_policy = fallback_policy

        self._last_rule: Rule | None = None
        self._last_predicates: dict[str, Any] | None = None
        self._last_action: int | None = None
        self._last_decision_source: str | None = None

    @property
    def rule_store(self) -> RuleStore:
        return self._rule_store

    @property
    def last_rule(self) -> Rule | None:
        """
        Правило, использованное при последнем решении.
        """
        return self._last_rule

    @property
    def last_predicates(self) -> dict[str, Any] | None:
        """
        Предикаты последнего обработанного состояния.
        """
        if self._last_predicates is None:
            return None

        return dict(self._last_predicates)

    @property
    def last_action(self) -> int | None:
        return self._last_action

    @property
    def last_decision_source(self) -> str | None:
        """
        Источник последнего действия:

        - rule;
        - fallback_policy;
        - fallback_action;
        - none.
        """
        return self._last_decision_source

    def choose_action(
        self,
        state: State,
    ) -> int | None:
        """
        Выбирает действие для переданного состояния.
        """
        predicates = self._predicate_generator.generate(state)

        self._last_predicates = dict(predicates)

        rule = self._rule_store.select_best(predicates)

        if rule is not None:
            self._last_rule = rule
            self._last_action = rule.action
            self._last_decision_source = "rule"

            return rule.action

        self._last_rule = None

        return self._choose_fallback(state)

    def find_matching_rules(
        self,
        state: State,
    ) -> list[Rule]:
        """
        Возвращает все правила, подходящие состоянию.
        """
        predicates = self._predicate_generator.generate(state)

        return self._rule_store.find_matching(predicates)

    def explain_decision(self) -> dict[str, Any]:
        """
        Возвращает объяснение последнего решения агента.
        """
        rule_data = (
            self._last_rule.to_dict()
            if self._last_rule is not None
            else None
        )

        return {
            "action": self._last_action,
            "source": self._last_decision_source,
            "predicates": (
                dict(self._last_predicates)
                if self._last_predicates is not None
                else None
            ),
            "rule": rule_data,
        }

    def reset_decision_history(self) -> None:
        """
        Очищает сведения о последнем решении.
        """
        self._last_rule = None
        self._last_predicates = None
        self._last_action = None
        self._last_decision_source = None

    def _choose_fallback(
        self,
        state: State,
    ) -> int | None:
        """
        Выбирает резервное действие,
        если подходящего правила нет.
        """
        if self._fallback_policy is not None:
            action = self._fallback_policy(state)

            if action is not None and not isinstance(action, int):
                raise TypeError(
                    "fallback_policy должна возвращать int или None."
                )

            self._last_action = action
            self._last_decision_source = "fallback_policy"

            return action

        if self._fallback_action is not None:
            self._last_action = self._fallback_action
            self._last_decision_source = "fallback_action"

            return self._fallback_action

        self._last_action = None
        self._last_decision_source = "none"

        return None