from __future__ import annotations

from copy import deepcopy
from typing import Any

from .transformation_rule import TransformationRule


class RulePredictor:
    """
    Применяет найденные TransformationRule
    к текущему состоянию мира.

    Идея:

        state + action
            ↓
        подходящие rules
            ↓
        predicted_state
    """

    def find_matching_rules(
        self,
        state: dict[str, Any],
        action: Any,
        rules: list[TransformationRule],
    ) -> list[TransformationRule]:

        matching = []

        for rule in rules:

            if rule.action != action:
                continue

            if not self._conditions_match(
                state,
                rule.conditions,
            ):
                continue

            matching.append(rule)

        return sorted(
            matching,
            key=lambda rule: (
                rule.confidence,
                rule.observations,
            ),
            reverse=True,
        )

    def predict(
        self,
        state: dict[str, Any],
        action: Any,
        rules: list[TransformationRule],
    ) -> dict[str, Any]:

        predicted_state = deepcopy(state)

        matching_rules = self.find_matching_rules(
            state=state,
            action=action,
            rules=rules,
        )

        for rule in matching_rules:

            predicted_state = self._apply_effects(
                predicted_state,
                rule.effects,
            )

        return predicted_state

    def predict_effects(
        self,
        state: dict[str, Any],
        action: Any,
        rules: list[TransformationRule],
    ) -> dict[str, Any]:

        matching_rules = self.find_matching_rules(
            state=state,
            action=action,
            rules=rules,
        )

        combined_effects = {}

        for rule in matching_rules:

            for feature, effect in rule.effects.items():

                if feature not in combined_effects:
                    combined_effects[feature] = effect

        return combined_effects

    @staticmethod
    def _conditions_match(
        state: dict[str, Any],
        conditions: dict[str, Any],
    ) -> bool:

        for feature, expected_value in conditions.items():

            if feature not in state:
                return False

            if state[feature] != expected_value:
                return False

        return True

    @staticmethod
    def _apply_effects(
        state: dict[str, Any],
        effects: dict[str, Any],
    ) -> dict[str, Any]:

        result = deepcopy(state)

        for feature, effect in effects.items():

            if not isinstance(effect, tuple):
                continue

            if len(effect) != 2:
                continue

            effect_type, value = effect

            # --------------------------------------------
            # DELTA
            #
            # x = 5
            # effect = ("delta", 1)
            #
            # x -> 6
            # --------------------------------------------

            if effect_type == "delta":

                current_value = result.get(
                    feature,
                    0,
                )

                if (
                    isinstance(current_value, (int, float))
                    and not isinstance(current_value, bool)
                ):

                    result[feature] = (
                        current_value + value
                    )

            # --------------------------------------------
            # SET
            #
            # message = ""
            # effect = ("set", "It's solid stone.")
            # --------------------------------------------

            elif effect_type == "set":

                result[feature] = value

        return result