from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable
from random import Random
from typing import Any

from src.state import State


StateKey = Hashable
StateEncoder = Callable[[State], StateKey]


class QTablePolicy:
    """
    Табличная Q-learning политика.

    Хранит оценки полезности действий:

        Q(state, action)

    Может:
    - выбирать действие;
    - исследовать среду через epsilon-greedy;
    - обновлять Q-значения;
    - работать как RL-политика для HybridAgent.
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
        normalized_actions = tuple(actions)

        if not normalized_actions:
            raise ValueError(
                "Список actions не может быть пустым."
            )

        if any(
            not isinstance(action, int)
            for action in normalized_actions
        ):
            raise TypeError(
                "Все действия должны иметь тип int."
            )

        if len(set(normalized_actions)) != len(normalized_actions):
            raise ValueError(
                "Список actions не должен содержать дубликаты."
            )

        if not 0.0 < learning_rate <= 1.0:
            raise ValueError(
                "learning_rate должен быть больше 0 и не больше 1."
            )

        if not 0.0 <= discount_factor <= 1.0:
            raise ValueError(
                "discount_factor должен находиться от 0 до 1."
            )

        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(
                "epsilon должен находиться от 0 до 1."
            )

        if (
            state_encoder is not None
            and not callable(state_encoder)
        ):
            raise TypeError(
                "state_encoder должен быть вызываемым объектом."
            )

        self._actions = normalized_actions
        self._learning_rate = float(learning_rate)
        self._discount_factor = float(discount_factor)
        self._epsilon = float(epsilon)

        self._state_encoder = (
            state_encoder or self._default_state_encoder
        )

        self._random = Random(seed)

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
                "epsilon должен находиться от 0 до 1."
            )

        self._epsilon = float(value)

    @property
    def state_count(self) -> int:
        """
        Количество известных состояний.
        """
        return len(self._q_table)

    def encode_state(
        self,
        state: State,
    ) -> StateKey:
        """
        Преобразует State в ключ Q-таблицы.
        """
        if not isinstance(state, State):
            raise TypeError(
                "state должен быть объектом State."
            )

        key = self._state_encoder(state)

        try:
            hash(key)
        except TypeError as error:
            raise TypeError(
                "state_encoder должен возвращать "
                "хешируемое значение."
            ) from error

        return key

    def ensure_state(
        self,
        state: State,
    ) -> StateKey:
        """
        Добавляет состояние в таблицу, если его ещё нет.

        Все новые действия получают Q-значение 0.
        """
        state_key = self.encode_state(state)

        if state_key not in self._q_table:
            self._q_table[state_key] = {
                action: 0.0
                for action in self._actions
            }

        return state_key

    def get_q_value(
        self,
        state: State,
        action: int,
    ) -> float:
        """
        Возвращает Q(state, action).
        """
        self._validate_action(action)

        state_key = self.ensure_state(state)

        return self._q_table[state_key][action]

    def set_q_value(
        self,
        state: State,
        action: int,
        value: float,
    ) -> None:
        """
        Принудительно устанавливает Q-значение.

        Полезно для тестов, загрузки модели и отладки.
        """
        self._validate_action(action)

        if not isinstance(value, (int, float)):
            raise TypeError(
                "value должен быть числом."
            )

        state_key = self.ensure_state(state)

        self._q_table[state_key][action] = float(value)

    def q_values(
        self,
        state: State,
    ) -> dict[int, float]:
        """
        Возвращает копию Q-значений состояния.
        """
        state_key = self.ensure_state(state)

        return dict(self._q_table[state_key])

    def best_actions(
        self,
        state: State,
    ) -> list[int]:
        """
        Возвращает все действия с максимальным Q-значением.

        Несколько действий могут иметь одинаковую оценку.
        """
        state_key = self.ensure_state(state)
        action_values = self._q_table[state_key]

        maximum_value = max(action_values.values())

        return [
            action
            for action, value in action_values.items()
            if value == maximum_value
        ]

    def best_action(
        self,
        state: State,
    ) -> int:
        """
        Выбирает одно из лучших действий.

        При одинаковых Q-значениях выбор случайный.
        """
        candidates = self.best_actions(state)

        return self._random.choice(candidates)

    def choose_action(
        self,
        state: State,
        explore: bool = True,
    ) -> int:
        """
        Выбирает действие по epsilon-greedy стратегии.

        explore=True:
            с вероятностью epsilon выполняется
            случайное действие.

        explore=False:
            всегда выбирается лучшее действие.
        """
        if not isinstance(explore, bool):
            raise TypeError(
                "explore должен иметь тип bool."
            )

        self.ensure_state(state)

        should_explore = (
            explore
            and self._random.random() < self._epsilon
        )

        if should_explore:
            return self._random.choice(self._actions)

        return self.best_action(state)

    def __call__(
        self,
        state: State,
    ) -> int:
        """
        Позволяет передавать объект как функцию:

            HybridAgent(
                rule_agent=rule_agent,
                rl_policy=q_policy,
            )
        """
        return self.choose_action(
            state,
            explore=True,
        )

    def update(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        done: bool = False,
    ) -> float:
        """
        Выполняет один шаг Q-learning.

        Формула:

        Q(s, a) ← Q(s, a) + α[
            r + γ max Q(s', a') - Q(s, a)
        ]

        Если done=True, будущая награда не учитывается.

        Возвращает новое Q-значение.
        """
        self._validate_action(action)

        if not isinstance(reward, (int, float)):
            raise TypeError(
                "reward должен быть числом."
            )

        if not isinstance(done, bool):
            raise TypeError(
                "done должен иметь тип bool."
            )

        state_key = self.ensure_state(state)
        next_state_key = self.ensure_state(next_state)

        current_q = self._q_table[state_key][action]

        if done:
            target = float(reward)
        else:
            next_max_q = max(
                self._q_table[next_state_key].values()
            )

            target = (
                float(reward)
                + self._discount_factor * next_max_q
            )

        new_q = current_q + self._learning_rate * (
            target - current_q
        )

        self._q_table[state_key][action] = new_q

        return new_q

    def decay_epsilon(
        self,
        decay_rate: float,
        minimum_epsilon: float = 0.01,
    ) -> float:
        """
        Уменьшает вероятность исследования.

        Новое значение:

            epsilon = max(
                minimum_epsilon,
                epsilon * decay_rate
            )
        """
        if not 0.0 <= decay_rate <= 1.0:
            raise ValueError(
                "decay_rate должен находиться от 0 до 1."
            )

        if not 0.0 <= minimum_epsilon <= 1.0:
            raise ValueError(
                "minimum_epsilon должен находиться от 0 до 1."
            )

        self._epsilon = max(
            float(minimum_epsilon),
            self._epsilon * float(decay_rate),
        )

        return self._epsilon

    def clear(self) -> None:
        """
        Полностью очищает Q-таблицу.
        """
        self._q_table.clear()

    def to_dict(self) -> dict[str, Any]:
        """
        Возвращает сериализуемое представление политики.

        Ключи состояний преобразуются в строки,
        потому что JSON не поддерживает tuple-ключи.
        """
        table_data = {
            repr(state_key): dict(action_values)
            for state_key, action_values in self._q_table.items()
        }

        return {
            "actions": list(self._actions),
            "learning_rate": self._learning_rate,
            "discount_factor": self._discount_factor,
            "epsilon": self._epsilon,
            "state_count": self.state_count,
            "q_table": table_data,
        }

    def _validate_action(
        self,
        action: int,
    ) -> None:
        if not isinstance(action, int):
            raise TypeError(
                "action должен иметь тип int."
            )

        if action not in self._actions:
            raise ValueError(
                f"Неизвестное действие: {action}."
            )

    @staticmethod
    def _default_state_encoder(
        state: State,
    ) -> StateKey:
        """
        Стандартный кодировщик состояния.

        Преобразует словарь State в отсортированный tuple.
        Это позволяет использовать состояние как ключ словаря.
        """
        state_data = state.to_dict()

        return tuple(
            sorted(state_data.items())
        )