from __future__ import annotations

import random
from typing import Iterable

from src.transformations.causal_link import CausalLink
from src.transformations.transformation_rule import TransformationRule


class CausalAgent:
    """
    Агент, использующий найденные TransformationRule
    и причинные цепочки между ними.

    Пока стратегия простая:

        1. найти правила, условия которых подходят текущему состоянию;
        2. предпочесть правило, которое входит в причинную цепочку;
        3. выбрать action этого правила;
        4. если подходящего правила нет — исследовать случайно.

    Это первая версия агента.
    """

    def __init__(
        self,
        actions: Iterable[int],
        rules: Iterable[TransformationRule],
        causal_links: Iterable[CausalLink] | None = None,
        epsilon: float = 0.15,
        seed: int | None = None,
    ):
        self.actions = list(actions)
        self.rules = list(rules)

        self.causal_links = list(
            causal_links or []
        )

        self.epsilon = epsilon

        self.random = random.Random(
            seed
        )

        # --------------------------------------------------------
        # Какие правила являются началом причинной связи
        # --------------------------------------------------------

        self.outgoing_links = {}

        for link in self.causal_links:

            rule_id = id(
                link.cause_rule
            )

            self.outgoing_links.setdefault(
                rule_id,
                [],
            )

            self.outgoing_links[
                rule_id
            ].append(link)

    # ============================================================
    # ВЫБОР ДЕЙСТВИЯ
    # ============================================================

    def choose_action(
        self,
        state,
    ) -> int:

        # --------------------------------------------------------
        # Исследование
        # --------------------------------------------------------

        if self.random.random() < self.epsilon:

            return self.random.choice(
                self.actions
            )

        # --------------------------------------------------------
        # Найти применимые правила
        # --------------------------------------------------------

        applicable_rules = []

        for rule in self.rules:

            if self._rule_matches(
                rule=rule,
                state=state,
            ):
                applicable_rules.append(
                    rule
                )

        # --------------------------------------------------------
        # Если ничего не знаем — случайное действие
        # --------------------------------------------------------

        if not applicable_rules:

            return self.random.choice(
                self.actions
            )

        # --------------------------------------------------------
        # Сначала предпочитаем правила,
        # которые продолжают причинную цепочку
        # --------------------------------------------------------

        chain_rules = []

        for rule in applicable_rules:

            if id(rule) in self.outgoing_links:

                chain_rules.append(
                    rule
                )

        if chain_rules:

            best_rule = max(
                chain_rules,
                key=self._rule_score,
            )

            return best_rule.action

        # --------------------------------------------------------
        # Иначе используем лучшее обычное правило
        # --------------------------------------------------------

        best_rule = max(
            applicable_rules,
            key=self._rule_score,
        )

        return best_rule.action

    # ============================================================
    # ПРОВЕРКА УСЛОВИЙ ПРАВИЛА
    # ============================================================

    def _rule_matches(
        self,
        rule: TransformationRule,
        state,
    ) -> bool:

        conditions = getattr(
            rule,
            "conditions",
            {},
        ) or {}

        # --------------------------------------------------------
        # Получаем атрибуты WorldState
        # --------------------------------------------------------

        state_attributes = getattr(
            state,
            "attributes",
            None,
        )

        if state_attributes is None:

            if isinstance(state, dict):
                state_attributes = state

            else:
                state_attributes = vars(
                    state
                )

        # --------------------------------------------------------
        # Проверяем каждое условие
        # --------------------------------------------------------

        for feature, required_value in conditions.items():

            if feature not in state_attributes:
                return False

            actual_value = state_attributes[
                feature
            ]

            if actual_value != required_value:
                return False

        return True

    # ============================================================
    # ОЦЕНКА ПРАВИЛА
    # ============================================================

    def _rule_score(
        self,
        rule: TransformationRule,
    ) -> float:

        confidence = getattr(
            rule,
            "confidence",
            0.0,
        )

        observations = getattr(
            rule,
            "observations",
            0,
        )

        try:
            confidence = float(
                confidence
            )

        except (TypeError, ValueError):
            confidence = 0.0

        try:
            observations = int(
                observations
            )

        except (TypeError, ValueError):
            observations = 0

        # confidence важнее,
        # observations используется как небольшой бонус

        return (
            confidence
            + min(
                observations,
                100,
            ) * 0.001
        )