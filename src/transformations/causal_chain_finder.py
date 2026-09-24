from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from src.transformations.causal_link import CausalLink
from src.transformations.transformation_rule import TransformationRule


class CausalChainFinder:
    """
    Ищет потенциальные причинные связи между TransformationRule.

    Главная идея:

        если эффект Rule A изменяет свойство,
        которое используется как условие Rule B,

    то A потенциально может создавать условия
    для применения B.

    Например:

        Rule A:
            effects = {
                "has_key": ("set", True)
            }

        Rule B:
            conditions = {
                "has_key": True
            }

    Тогда:

        Rule A
            ↓
        has_key = True
            ↓
        Rule B

    Это пока структурный поиск причинных связей.

    Позже эти связи можно дополнительно подтверждать
    реальными последовательностями TransitionMemory.
    """

    def __init__(
        self,
        minimum_confidence: float = 0.0,
    ):
        self.minimum_confidence = minimum_confidence

    # ============================================================
    # ОСНОВНОЙ ПОИСК
    # ============================================================

    def find_links(
        self,
        rules: Iterable[TransformationRule],
    ) -> list[CausalLink]:

        rules = list(rules)

        links: list[CausalLink] = []

        for cause_rule in rules:

            for effect_rule in rules:

                # правило не связываем само с собой
                if cause_rule is effect_rule:
                    continue

                pair_links = self._compare_rules(
                    cause_rule=cause_rule,
                    effect_rule=effect_rule,
                )

                for link in pair_links:

                    if link.confidence >= self.minimum_confidence:
                        links.append(link)

        return links

    # ============================================================
    # СРАВНЕНИЕ ДВУХ ПРАВИЛ
    # ============================================================

    def _compare_rules(
        self,
        cause_rule: TransformationRule,
        effect_rule: TransformationRule,
    ) -> list[CausalLink]:

        links: list[CausalLink] = []

        cause_effects = getattr(
            cause_rule,
            "effects",
            {},
        ) or {}

        effect_conditions = getattr(
            effect_rule,
            "conditions",
            {},
        ) or {}

        for feature, effect in cause_effects.items():

            if feature not in effect_conditions:
                continue

            required_value = effect_conditions[feature]

            confidence = self._calculate_structural_confidence(
                effect=effect,
                required_value=required_value,
            )

            if confidence <= 0:
                continue

            observations = min(
                self._get_observations(cause_rule),
                self._get_observations(effect_rule),
            )

            link = CausalLink(
                cause_rule=cause_rule,
                effect_rule=effect_rule,
                connecting_feature=feature,
                confidence=confidence,
                observations=observations,
            )

            links.append(link)

        return links

    # ============================================================
    # ОЦЕНКА СТРУКТУРНОЙ СВЯЗИ
    # ============================================================

    def _calculate_structural_confidence(
        self,
        effect,
        required_value,
    ) -> float:
        """
        Оценивает, насколько эффект первого правила
        способен создать условие второго правила.

        Поддерживаем эффекты:

            ("set", value)
            ("delta", value)

        Для set связь может быть установлена напрямую.

        Для delta без исходного состояния невозможно точно
        определить конечное значение, поэтому пока такая связь
        считается более слабой.
        """

        if not isinstance(effect, tuple):
            return 0.0

        if len(effect) != 2:
            return 0.0

        effect_type, effect_value = effect

        # --------------------------------------------------------
        # SET
        # --------------------------------------------------------

        if effect_type == "set":

            if effect_value == required_value:
                return 1.0

            return 0.0

        # --------------------------------------------------------
        # DELTA
        # --------------------------------------------------------

        if effect_type == "delta":
            """
            Например:

                Rule A:
                    x += 1

                Rule B:
                    IF x == 10

            Мы пока не знаем x ДО применения A.

            Поэтому нельзя утверждать:

                x += 1 -> x == 10

            Но свойства связаны структурно.

            Реальную уверенность позже даст TransitionMemory.
            """

            return 0.25

        return 0.0

    # ============================================================
    # ЧИСЛО НАБЛЮДЕНИЙ
    # ============================================================

    @staticmethod
    def _get_observations(
        rule: TransformationRule,
    ) -> int:

        value = getattr(
            rule,
            "observations",
            0,
        )

        try:
            return int(value)

        except (TypeError, ValueError):
            return 0

    # ============================================================
    # ГРАФ СВЯЗЕЙ
    # ============================================================

    def build_graph(
        self,
        links: Iterable[CausalLink],
    ):
        """
        Создаёт простой ориентированный граф:

            rule -> [links...]

        Пока без NetworkX.

        Это обычный словарь.
        """

        graph = defaultdict(list)

        for link in links:

            graph[id(link.cause_rule)].append(link)

        return dict(graph)

    # ============================================================
    # ПОИСК ЦЕПОЧЕК
    # ============================================================

    def find_chains(
        self,
        links: Iterable[CausalLink],
        max_depth: int = 5,
    ) -> list[list[CausalLink]]:
        """
        Ищет цепочки:

            Rule A
                ↓
            Rule B
                ↓
            Rule C

        max_depth:
            максимальное количество причинных связей
            внутри одной цепочки.
        """

        links = list(links)

        outgoing = defaultdict(list)

        for link in links:

            outgoing[id(link.cause_rule)].append(link)

        chains: list[list[CausalLink]] = []

        for start_link in links:

            self._expand_chain(
                current_chain=[start_link],
                outgoing=outgoing,
                chains=chains,
                max_depth=max_depth,
                visited_rule_ids={
                    id(start_link.cause_rule),
                    id(start_link.effect_rule),
                },
            )

        return chains

    # ============================================================
    # РЕКУРСИВНОЕ РАСШИРЕНИЕ ЦЕПОЧКИ
    # ============================================================

    def _expand_chain(
        self,
        current_chain: list[CausalLink],
        outgoing,
        chains: list[list[CausalLink]],
        max_depth: int,
        visited_rule_ids: set[int],
    ):
        """
        Продолжает цепочку от последнего правила.
        """

        chains.append(
            list(current_chain)
        )

        if len(current_chain) >= max_depth:
            return

        last_link = current_chain[-1]

        current_rule = last_link.effect_rule

        next_links = outgoing.get(
            id(current_rule),
            [],
        )

        for next_link in next_links:

            next_rule_id = id(
                next_link.effect_rule
            )

            # пока запрещаем циклы
            if next_rule_id in visited_rule_ids:
                continue

            new_visited = set(
                visited_rule_ids
            )

            new_visited.add(
                next_rule_id
            )

            self._expand_chain(
                current_chain=(
                    current_chain
                    + [next_link]
                ),
                outgoing=outgoing,
                chains=chains,
                max_depth=max_depth,
                visited_rule_ids=new_visited,
            )