from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.agents.rule_agent import RuleAgent
from src.rules import Rule
from src.state import State


RLPolicy = Callable[[State], int]


class HybridAgent:
    """
    Гибридный агент.

    Сначала пытается выбрать действие по символическим правилам.
    Если подходящего правила нет, использует RL-политику.

    Благодаря этому:
    - правила дают объяснимые решения;
    - RL действует в неизвестных ситуациях;
    - позже успешный RL-опыт можно превращать в новые правила.
    """

    def __init__(
        self,
        rule_agent: RuleAgent,
        rl_policy: RLPolicy,
    ) -> None:
        if not isinstance(rule_agent, RuleAgent):
            raise TypeError(
                "rule_agent должен быть объектом RuleAgent."
            )

        if not callable(rl_policy):
            raise TypeError(
                "rl_policy должна быть вызываемым объектом."
            )

        self._rule_agent = rule_agent
        self._rl_policy = rl_policy

        self._last_action: int | None = None
        self._last_source: str | None = None
        self._last_rule: Rule | None = None
        self._last_state: State | None = None

    @property
    def rule_agent(self) -> RuleAgent:
        return self._rule_agent

    @property
    def last_action(self) -> int | None:
        return self._last_action

    @property
    def last_source(self) -> str | None:
        """
        Источник последнего решения:

        - rule;
        - rl.
        """
        return self._last_source

    @property
    def last_rule(self) -> Rule | None:
        return self._last_rule

    @property
    def last_state(self) -> State | None:
        return self._last_state

    def choose_action(
        self,
        state: State,
    ) -> int:
        """
        Выбирает действие.

        Приоритет:
        1. подходящее символическое правило;
        2. RL-политика.
        """
        if not isinstance(state, State):
            raise TypeError(
                "state должен быть объектом State."
            )

        self._last_state = state

        rule_action = self._rule_agent.choose_action(state)

        if (
            rule_action is not None
            and self._rule_agent.last_decision_source == "rule"
        ):
            self._last_action = rule_action
            self._last_source = "rule"
            self._last_rule = self._rule_agent.last_rule

            return rule_action

        rl_action = self._rl_policy(state)

        if not isinstance(rl_action, int):
            raise TypeError(
                "rl_policy должна возвращать int."
            )

        self._last_action = rl_action
        self._last_source = "rl"
        self._last_rule = None

        return rl_action

    def explain_decision(self) -> dict[str, Any]:
        """
        Возвращает объяснение последнего решения.
        """
        return {
            "action": self._last_action,
            "source": self._last_source,
            "state": (
                self._last_state.to_dict()
                if self._last_state is not None
                else None
            ),
            "rule": (
                self._last_rule.to_dict()
                if self._last_rule is not None
                else None
            ),
            "predicates": self._rule_agent.last_predicates,
        }

    def reset_decision_history(self) -> None:
        """
        Очищает историю последнего решения.
        """
        self._last_action = None
        self._last_source = None
        self._last_rule = None
        self._last_state = None

        self._rule_agent.reset_decision_history()