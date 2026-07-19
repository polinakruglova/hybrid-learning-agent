from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from src.memory import Experience
from src.predicates import PredicateGenerator
from src.rules.rule import Rule
from src.rules.store import RuleStore


DEFAULT_PREDICATE_KEYS: tuple[str, ...] = (
    "has_key",
    "door_open",
    "door_locked",
    "key_visible",
    "door_visible",
    "goal_visible",
    "near_key",
    "near_door",
    "near_goal",
    "key_left",
    "key_right",
    "key_above",
    "key_below",
    "door_left",
    "door_right",
    "door_above",
    "door_below",
    "goal_left",
    "goal_right",
    "goal_above",
    "goal_below",
    "same_row_with_key",
    "same_column_with_key",
    "same_row_with_door",
    "same_column_with_door",
    "same_row_with_goal",
    "same_column_with_goal",
)


class RuleMiner:
    """
    Извлекает правила поведения из опыта агента.

    Правило создаётся только из перехода, который:

    - имеет награду выше minimum_reward;
    - помечен как успешный;
    - получен из разрешённого источника, если задан
      allowed_sources.
    """

    def __init__(
        self,
        predicate_generator: PredicateGenerator | None = None,
        selected_predicates: Iterable[str] | None = None,
        minimum_reward: float = 0.0,
        allowed_sources: Iterable[str] | None = None,
    ) -> None:
        self._predicate_generator = (
            predicate_generator or PredicateGenerator()
        )

        self._selected_predicates = tuple(
            selected_predicates or DEFAULT_PREDICATE_KEYS
        )

        self._minimum_reward = float(minimum_reward)

        self._allowed_sources = (
            set(allowed_sources)
            if allowed_sources is not None
            else None
        )

    def can_mine(
        self,
        experience: Experience,
    ) -> bool:
        """
        Проверяет, можно ли создать правило из опыта.
        """
        if not isinstance(experience, Experience):
            raise TypeError(
                "experience должен быть объектом Experience."
            )

        if experience.reward <= self._minimum_reward:
            return False

        if not experience.success:
            return False

        if (
            self._allowed_sources is not None
            and experience.source not in self._allowed_sources
        ):
            return False

        return True

    def mine(
        self,
        experience: Experience,
    ) -> Rule | None:
        """
        Создаёт одно правило из успешного перехода.

        Если переход не подходит, возвращает None.
        """
        if not self.can_mine(experience):
            return None

        predicates = self._predicate_generator.generate(
            experience.state
        )

        conditions = self._select_conditions(
            predicates
        )

        if not conditions:
            return None

        return Rule(
            conditions=conditions,
            action=experience.action,
            support=1,
            successes=1,
            total_reward=float(experience.reward),
            metadata={
                "source": experience.source,
                "mined": True,
            },
        )

    def mine_many(
        self,
        experiences: Iterable[Experience],
    ) -> list[Rule]:
        """
        Создаёт правила из набора переходов.
        """
        mined_rules: list[Rule] = []

        for experience in experiences:
            rule = self.mine(experience)

            if rule is not None:
                mined_rules.append(rule)

        return mined_rules

    def mine_into_store(
        self,
        experiences: Iterable[Experience],
        store: RuleStore,
    ) -> list[Rule]:
        """
        Создаёт правила и добавляет их в RuleStore.

        Возвращает только реально добавленные правила.
        """
        if not isinstance(store, RuleStore):
            raise TypeError(
                "store должен быть объектом RuleStore."
            )

        added_rules: list[Rule] = []

        for experience in experiences:
            candidate = self.mine(experience)

            if candidate is None:
                continue

            if store.contains_equivalent(candidate):
                continue

            store.add(candidate)
            added_rules.append(candidate)

        return added_rules

    def _select_conditions(
        self,
        predicates: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Оставляет только выбранные признаки.

        None не включается в правило, потому что это
        означает неизвестное значение.
        """
        conditions: dict[str, Any] = {}

        for name in self._selected_predicates:
            if name not in predicates:
                continue

            value = predicates[name]

            if value is None:
                continue

            conditions[name] = value

        return conditions