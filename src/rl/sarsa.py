from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Any, Callable

from src.state import State


StateKey = tuple[Any, ...]
StateEncoder = Callable[[State], StateKey]


class SARSAPolicy:
    """
    Табличная on-policy SARSA-политика.

    Формула обновления:

        Q(s, a) <- Q(s, a) + alpha *
            [r + gamma * Q(s_next, a_next) - Q(s, a)]

    В отличие от Q-learning, SARSA обновляет значение через действие,
    которое агент действительно выбрал в следующем состоянии.
    """

    def __init__(
        self,
        actions: Iterable[int],
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 0.1,
        state_encoder: StateEncoder | None = None,
        seed: int | None = None,
    ) -> None:
        self._actions = tuple(actions)

        if not self._actions:
            raise ValueError(
                "actions must contain at least one action"
            )

        if not 0.0 < learning_rate <= 1.0:
            raise ValueError(
                "learning_rate must be in the interval (0, 1]"
            )

        if not 0.0 <= discount_factor <= 1.0:
            raise ValueError(
                "discount_factor must be in the interval [0, 1]"
            )

        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(
                "epsilon must be in the interval [0, 1]"
            )

        self._learning_rate = float(learning_rate)
        self._discount_factor = float(discount_factor)
        self._epsilon = float(epsilon)

        self._state_encoder = (
            state_encoder
            if state_encoder is not None
            else self._default_state_encoder
        )

        self._random = random.Random(seed)

        self._q_table: dict[
            StateKey,
            dict[int, float],
        ] = {}

    @property
    def actions(self) -> tuple[int, ...]:
        return self._actions

    @property
    def learning_rate(self) -> float:
        return self._learning_rate

    @property
    def discount_factor(self) -> float:
        return self._discount_factor

    @property
    def epsilon(self) -> float:
        return self._epsilon

    @epsilon.setter
    def epsilon(self, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "epsilon must be in the interval [0, 1]"
            )

        self._epsilon = float(value)

    @property
    def state_count(self) -> int:
        return len(self._q_table)

    def encode_state(
        self,
        state: State,
    ) -> StateKey:
        return self._state_encoder(state)

    def ensure_state(
        self,
        state: State,
    ) -> StateKey:
        state_key = self.encode_state(state)

        if state_key not in self._q_table:
            self._q_table[state_key] = {
                action: 0.0
                for action in self._actions
            }

        return state_key

    def q_values(
        self,
        state: State,
    ) -> dict[int, float]:
        state_key = self.ensure_state(state)

        return dict(self._q_table[state_key])

    def get_q_value(
        self,
        state: State,
        action: int,
    ) -> float:
        self._validate_action(action)

        state_key = self.ensure_state(state)

        return self._q_table[state_key][action]

    def set_q_value(
        self,
        state: State,
        action: int,
        value: float,
    ) -> None:
        self._validate_action(action)

        state_key = self.ensure_state(state)

        self._q_table[state_key][action] = float(value)

    def best_actions(
        self,
        state: State,
    ) -> list[int]:
        state_key = self.ensure_state(state)
        values = self._q_table[state_key]

        maximum_value = max(values.values())

        return [
            action
            for action, value in values.items()
            if value == maximum_value
        ]

    def best_action(
        self,
        state: State,
    ) -> int:
        return self._random.choice(
            self.best_actions(state)
        )

    def choose_action(
        self,
        state: State,
        explore: bool = True,
    ) -> int:
        self.ensure_state(state)

        should_explore = (
            explore
            and self._random.random() < self._epsilon
        )

        if should_explore:
            return self._random.choice(
                self._actions
            )

        return self.best_action(state)

    def update(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        next_action: int | None,
        done: bool = False,
    ) -> float:
        """
        Выполняет одно SARSA-обновление.

        Возвращает новое значение Q(state, action).
        """

        self._validate_action(action)

        if not done and next_action is None:
            raise ValueError(
                "next_action is required when done is False"
            )

        if next_action is not None:
            self._validate_action(next_action)

        state_key = self.ensure_state(state)
        old_value = self._q_table[state_key][action]

        if done:
            target = float(reward)
        else:
            next_state_key = self.ensure_state(
                next_state
            )

            next_value = self._q_table[
                next_state_key
            ][next_action]

            target = (
                float(reward)
                + self._discount_factor
                * next_value
            )

        new_value = (
            old_value
            + self._learning_rate
            * (target - old_value)
        )

        self._q_table[state_key][action] = new_value

        return new_value

    def decay_epsilon(
        self,
        decay_rate: float,
        minimum_epsilon: float = 0.01,
    ) -> float:
        if not 0.0 < decay_rate <= 1.0:
            raise ValueError(
                "decay_rate must be in the interval (0, 1]"
            )

        if not 0.0 <= minimum_epsilon <= 1.0:
            raise ValueError(
                "minimum_epsilon must be in the interval [0, 1]"
            )

        self._epsilon = max(
            minimum_epsilon,
            self._epsilon * decay_rate,
        )

        return self._epsilon

    def clear(self) -> None:
        self._q_table.clear()

    def to_dict(self) -> dict[str, Any]:
        states: list[dict[str, Any]] = []

        for state_key, action_values in (
            self._q_table.items()
        ):
            states.append(
                {
                    "state": list(state_key),
                    "actions": {
                        str(action): value
                        for action, value
                        in action_values.items()
                    },
                }
            )

        return {
            "algorithm": "sarsa",
            "learning_rate": self._learning_rate,
            "discount_factor": (
                self._discount_factor
            ),
            "epsilon": self._epsilon,
            "actions": list(self._actions),
            "state_count": self.state_count,
            "states": states,
        }

    def _validate_action(
        self,
        action: int,
    ) -> None:
        if action not in self._actions:
            raise ValueError(
                f"unknown action: {action}"
            )

    @staticmethod
    def _default_state_encoder(
        state: State,
    ) -> StateKey:
        return (
            state.agent_position,
            state.agent_direction,
            state.key_position,
            state.door_position,
            state.goal_position,
            state.has_key,
            state.door_open,
            state.door_locked,
            state.key_visible,
            state.door_visible,
            state.goal_visible,
        )