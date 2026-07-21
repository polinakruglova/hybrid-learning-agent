from __future__ import annotations

from typing import Any

from src.agents import HybridAgent
from src.memory import Experience, ExperienceBuffer
from src.rl import QTablePolicy
from src.rules import RuleStore
from src.state import State


class LearningController:
    """
    Координатор обучения гибридного агента.

    Отвечает за:

    - выбор действия через HybridAgent;
    - сохранение опыта в ExperienceBuffer;
    - обновление Q-таблицы;
    - сбор статистики обучения.

    Символические правила здесь не создаются.

    Правила периодически извлекаются из накопленного опыта
    отдельным NeuralRuleTrainer.
    """

    def __init__(
        self,
        agent: HybridAgent,
        q_policy: QTablePolicy,
        experience_buffer: ExperienceBuffer,
        rule_store: RuleStore,
    ) -> None:
        if not isinstance(agent, HybridAgent):
            raise TypeError(
                "agent должен быть объектом HybridAgent."
            )

        if not isinstance(q_policy, QTablePolicy):
            raise TypeError(
                "q_policy должен быть объектом QTablePolicy."
            )

        if not isinstance(
            experience_buffer,
            ExperienceBuffer,
        ):
            raise TypeError(
                "experience_buffer должен быть объектом "
                "ExperienceBuffer."
            )

        if not isinstance(rule_store, RuleStore):
            raise TypeError(
                "rule_store должен быть объектом RuleStore."
            )

        self._agent = agent
        self._q_policy = q_policy
        self._experience_buffer = experience_buffer
        self._rule_store = rule_store

        self._total_steps = 0
        self._total_episodes = 0
        self._successful_episodes = 0
        self._total_reward = 0.0

        self._last_experience: Experience | None = None
        self._last_q_value: float | None = None

    @property
    def agent(self) -> HybridAgent:
        return self._agent

    @property
    def q_policy(self) -> QTablePolicy:
        return self._q_policy

    @property
    def experience_buffer(
        self,
    ) -> ExperienceBuffer:
        return self._experience_buffer

    @property
    def rule_store(self) -> RuleStore:
        return self._rule_store

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @property
    def total_episodes(self) -> int:
        return self._total_episodes

    @property
    def successful_episodes(self) -> int:
        return self._successful_episodes

    @property
    def total_reward(self) -> float:
        return self._total_reward

    @property
    def success_rate(self) -> float:
        if self._total_episodes == 0:
            return 0.0

        return (
            self._successful_episodes
            / self._total_episodes
        )

    @property
    def average_reward_per_step(self) -> float:
        if self._total_steps == 0:
            return 0.0

        return self._total_reward / self._total_steps

    @property
    def last_experience(
        self,
    ) -> Experience | None:
        return self._last_experience

    @property
    def last_q_value(self) -> float | None:
        return self._last_q_value

    def choose_action(
        self,
        state: State,
    ) -> int:
        """
        Выбирает действие через HybridAgent.
        """
        if not isinstance(state, State):
            raise TypeError(
                "state должен быть объектом State."
            )

        return self._agent.choose_action(state)

    def learn(
        self,
        state: State,
        action: int,
        reward: float,
        next_state: State,
        done: bool = False,
        success: bool = False,
        source: str | None = None,
    ) -> Experience:
        """
        Обрабатывает один переход среды.

        Переход сохраняется в ExperienceBuffer,
        после чего обновляется Q-таблица.

        Правила после отдельного перехода не создаются.
        """
        self._validate_transition(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            success=success,
            source=source,
        )

        experience_source = (
            source
            or self._agent.last_source
            or "unknown"
        )

        experience = Experience(
            state=state,
            action=action,
            reward=float(reward),
            next_state=next_state,
            done=done,
            success=success,
            source=experience_source,
        )

        # Сохраняем переход для последующего обучения
        # NeuralPatternMiner.
        self._experience_buffer.add(experience)

        # Обновляем Q-learning.
        new_q_value = self._q_policy.update(
            state=state,
            action=action,
            reward=float(reward),
            next_state=next_state,
            done=done,
        )

        self._total_steps += 1
        self._total_reward += float(reward)

        if done:
            self._total_episodes += 1

            if success:
                self._successful_episodes += 1

        self._last_experience = experience
        self._last_q_value = new_q_value

        return experience

    def step(
        self,
        state: State,
        reward: float,
        next_state: State,
        done: bool = False,
        success: bool = False,
    ) -> Experience:
        """
        Обрабатывает результат последнего действия агента.
        """
        action = self._agent.last_action

        if action is None:
            raise RuntimeError(
                "У агента нет последнего действия. "
                "Сначала вызови choose_action(state)."
            )

        return self.learn(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
            success=success,
            source=self._agent.last_source,
        )

    def decay_epsilon(
        self,
        decay_rate: float,
        minimum_epsilon: float = 0.01,
    ) -> float:
        """
        Уменьшает epsilon Q-политики.
        """
        return self._q_policy.decay_epsilon(
            decay_rate=decay_rate,
            minimum_epsilon=minimum_epsilon,
        )

    def statistics(self) -> dict[str, Any]:
        """
        Возвращает общую статистику обучения.
        """
        return {
            "total_steps": self._total_steps,
            "total_episodes": self._total_episodes,
            "successful_episodes": (
                self._successful_episodes
            ),
            "success_rate": self.success_rate,
            "total_reward": self._total_reward,
            "average_reward_per_step": (
                self.average_reward_per_step
            ),
            "experience_count": len(
                self._experience_buffer
            ),
            "rule_count": len(self._rule_store),
            "q_state_count": self._q_policy.state_count,
            "epsilon": self._q_policy.epsilon,
        }

    def reset_statistics(self) -> None:
        """
        Сбрасывает статистику контроллера.
        """
        self._total_steps = 0
        self._total_episodes = 0
        self._successful_episodes = 0
        self._total_reward = 0.0

        self._last_experience = None
        self._last_q_value = None

    def clear_learning_data(self) -> None:
        """
        Полностью очищает накопленные данные обучения.
        """
        self._experience_buffer.clear()
        self._q_policy.clear()
        self._rule_store.clear()

        self.reset_statistics()
        self._agent.reset_decision_history()

    @staticmethod
    def _validate_transition(
        state: State,
        action: int,
        reward: float,
        next_state: State,
        done: bool,
        success: bool,
        source: str | None,
    ) -> None:
        if not isinstance(state, State):
            raise TypeError(
                "state должен быть объектом State."
            )

        if not isinstance(next_state, State):
            raise TypeError(
                "next_state должен быть объектом State."
            )

        if not isinstance(action, int):
            raise TypeError(
                "action должен иметь тип int."
            )

        if not isinstance(reward, (int, float)):
            raise TypeError(
                "reward должен быть числом."
            )

        if not isinstance(done, bool):
            raise TypeError(
                "done должен иметь тип bool."
            )

        if not isinstance(success, bool):
            raise TypeError(
                "success должен иметь тип bool."
            )

        if (
            source is not None
            and not isinstance(source, str)
        ):
            raise TypeError(
                "source должен быть строкой или None."
            )

        if source == "":
            raise ValueError(
                "source не может быть пустой строкой."
            )