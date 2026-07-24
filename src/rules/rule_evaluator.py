from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Any

from src.memory import Experience
from src.predicates import PredicateGenerator
from src.rules.neural_pattern_miner import (
    NeuralRuleCandidate,
)


@dataclass(frozen=True, slots=True)
class EvaluatedTransition:
    """
    Переход с рассчитанным результатом до конца эпизода.

    return_to_go:
        Дисконтированная сумма наград от текущего шага
        до завершения эпизода.

    episode_success:
        Завершился ли соответствующий эпизод успешно.
    """

    experience: Experience
    predicates: dict[str, Any]
    return_to_go: float
    episode_success: bool


@dataclass(frozen=True, slots=True)
class RuleEvaluation:
    """
    Статистическая оценка гипотезы правила.
    """

    conditions: dict[str, Any]
    action: int

    # Сколько раз встретилось состояние с такими условиями.
    state_support: int

    # Сколько раз в таком состоянии было выполнено
    # именно действие кандидата.
    action_support: int

    # Доля выполнения этого действия среди всех действий
    # при данных условиях.
    action_frequency: float

    # Доля эпизодов, которые после этого действия
    # завершились успехом.
    episode_success_rate: float

    # Доля переходов с положительной непосредственной наградой.
    positive_reward_rate: float

    average_immediate_reward: float
    average_return_to_go: float

    # Средний return-to-go альтернативных действий
    # при тех же условиях.
    alternative_average_return: float | None

    # Насколько действие правила лучше альтернатив.
    return_advantage: float | None

    neural_confidence: float
    neural_support: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "conditions": dict(self.conditions),
            "action": self.action,
            "state_support": self.state_support,
            "action_support": self.action_support,
            "action_frequency": self.action_frequency,
            "episode_success_rate": (
                self.episode_success_rate
            ),
            "positive_reward_rate": (
                self.positive_reward_rate
            ),
            "average_immediate_reward": (
                self.average_immediate_reward
            ),
            "average_return_to_go": (
                self.average_return_to_go
            ),
            "alternative_average_return": (
                self.alternative_average_return
            ),
            "return_advantage": self.return_advantage,
            "neural_confidence": (
                self.neural_confidence
            ),
            "neural_support": self.neural_support,
        }


class RuleEvaluator:
    """
    Проверяет нейросетевые гипотезы на полном опыте агента.

    NeuralPatternMiner отвечает на вопрос:

        «Какая закономерность может существовать?»

    RuleEvaluator отвечает:

        «Подтверждается ли эта закономерность
        всей накопленной историей?»
    """

    def __init__(
        self,
        predicate_generator: PredicateGenerator,
        *,
        discount_factor: float = 0.95,
        minimum_state_support: int = 10,
        minimum_action_support: int = 5,
        minimum_episode_success_rate: float = 0.50,
        minimum_return_advantage: float = 0.0,
    ) -> None:
        if not isinstance(
            predicate_generator,
            PredicateGenerator,
        ):
            raise TypeError(
                "predicate_generator должен быть "
                "PredicateGenerator."
            )

        if not 0.0 <= discount_factor <= 1.0:
            raise ValueError(
                "discount_factor должен быть от 0 до 1."
            )

        if minimum_state_support <= 0:
            raise ValueError(
                "minimum_state_support должен быть "
                "положительным."
            )

        if minimum_action_support <= 0:
            raise ValueError(
                "minimum_action_support должен быть "
                "положительным."
            )

        if not 0.0 <= minimum_episode_success_rate <= 1.0:
            raise ValueError(
                "minimum_episode_success_rate должен быть "
                "от 0 до 1."
            )

        self.predicate_generator = predicate_generator
        self.discount_factor = float(discount_factor)

        self.minimum_state_support = int(
            minimum_state_support
        )
        self.minimum_action_support = int(
            minimum_action_support
        )
        self.minimum_episode_success_rate = float(
            minimum_episode_success_rate
        )
        self.minimum_return_advantage = float(
            minimum_return_advantage
        )

    def prepare_transitions(
        self,
        experiences: Iterable[Experience],
    ) -> list[EvaluatedTransition]:
        """
        Рассчитывает return-to-go и итоговый успех
        для каждого перехода.

        Порядок Experience должен соответствовать
        реальному порядку их появления в среде.
        """

        experience_list = list(experiences)

        for experience in experience_list:
            if not isinstance(experience, Experience):
                raise TypeError(
                    "Все элементы должны быть Experience."
                )

        if not experience_list:
            return []

        prepared_reversed: list[
            EvaluatedTransition
        ] = []

        future_return = 0.0
        future_success = False

        for experience in reversed(experience_list):
            if experience.done:
                current_return = float(
                    experience.reward
                )

                current_success = bool(
                    experience.success
                )
            else:
                current_return = (
                    float(experience.reward)
                    + self.discount_factor
                    * future_return
                )

                current_success = future_success

            predicates = (
                self.predicate_generator.generate(
                    experience.state
                )
            )

            prepared_reversed.append(
                EvaluatedTransition(
                    experience=experience,
                    predicates=predicates,
                    return_to_go=current_return,
                    episode_success=current_success,
                )
            )

            future_return = current_return
            future_success = current_success

        prepared_reversed.reverse()

        return prepared_reversed

    def evaluate(
        self,
        candidate: NeuralRuleCandidate,
        experiences: Iterable[Experience],
    ) -> RuleEvaluation:
        """
        Проверяет одного кандидата на всём опыте.
        """

        if not isinstance(
            candidate,
            NeuralRuleCandidate,
        ):
            raise TypeError(
                "candidate должен быть "
                "NeuralRuleCandidate."
            )

        prepared = self.prepare_transitions(
            experiences
        )

        state_matches = [
            transition
            for transition in prepared
            if self._conditions_match(
                conditions=candidate.conditions,
                predicates=transition.predicates,
            )
        ]

        action_matches = [
            transition
            for transition in state_matches
            if (
                transition.experience.action
                == candidate.action
            )
        ]

        alternative_matches = [
            transition
            for transition in state_matches
            if (
                transition.experience.action
                != candidate.action
            )
        ]

        state_support = len(state_matches)
        action_support = len(action_matches)

        if state_support == 0:
            action_frequency = 0.0
        else:
            action_frequency = (
                action_support / state_support
            )

        if action_support == 0:
            episode_success_rate = 0.0
            positive_reward_rate = 0.0
            average_immediate_reward = 0.0
            average_return_to_go = 0.0
        else:
            episode_success_rate = (
                sum(
                    transition.episode_success
                    for transition in action_matches
                )
                / action_support
            )

            positive_reward_rate = (
                sum(
                    transition.experience.reward > 0
                    for transition in action_matches
                )
                / action_support
            )

            average_immediate_reward = (
                sum(
                    transition.experience.reward
                    for transition in action_matches
                )
                / action_support
            )

            average_return_to_go = (
                sum(
                    transition.return_to_go
                    for transition in action_matches
                )
                / action_support
            )

        if alternative_matches:
            alternative_average_return = (
                sum(
                    transition.return_to_go
                    for transition in alternative_matches
                )
                / len(alternative_matches)
            )

            return_advantage = (
                average_return_to_go
                - alternative_average_return
            )
        else:
            alternative_average_return = None
            return_advantage = None

        return RuleEvaluation(
            conditions=dict(candidate.conditions),
            action=candidate.action,
            state_support=state_support,
            action_support=action_support,
            action_frequency=action_frequency,
            episode_success_rate=(
                episode_success_rate
            ),
            positive_reward_rate=(
                positive_reward_rate
            ),
            average_immediate_reward=(
                average_immediate_reward
            ),
            average_return_to_go=(
                average_return_to_go
            ),
            alternative_average_return=(
                alternative_average_return
            ),
            return_advantage=return_advantage,
            neural_confidence=candidate.confidence,
            neural_support=candidate.support,
        )

    def evaluate_many(
        self,
        candidates: Iterable[NeuralRuleCandidate],
        experiences: Iterable[Experience],
    ) -> list[RuleEvaluation]:
        """
        Проверяет несколько гипотез.
        """

        candidate_list = list(candidates)
        experience_list = list(experiences)

        return [
            self.evaluate(
                candidate=candidate,
                experiences=experience_list,
            )
            for candidate in candidate_list
        ]

    def accept(
        self,
        evaluation: RuleEvaluation,
    ) -> bool:
        """
        Решает, достаточно ли подтверждена гипотеза.
        """

        if not isinstance(
            evaluation,
            RuleEvaluation,
        ):
            raise TypeError(
                "evaluation должен быть RuleEvaluation."
            )

        if (
            evaluation.state_support
            < self.minimum_state_support
        ):
            return False

        if (
            evaluation.action_support
            < self.minimum_action_support
        ):
            return False

        if (
            evaluation.episode_success_rate
            < self.minimum_episode_success_rate
        ):
            return False

        if (
            evaluation.return_advantage is not None
            and evaluation.return_advantage
            < self.minimum_return_advantage
        ):
            return False

        return True

    def accepted(
        self,
        evaluations: Iterable[RuleEvaluation],
    ) -> list[RuleEvaluation]:
        """
        Возвращает только подтверждённые гипотезы.
        """

        return [
            evaluation
            for evaluation in evaluations
            if self.accept(evaluation)
        ]

    def select_best_conflicts(
        self,
        evaluations: Iterable[RuleEvaluation],
    ) -> list[RuleEvaluation]:
        """
        Разрешает конфликт одинаковых условий
        с разными действиями.

        Для каждого набора условий оставляется действие
        с лучшей статистической оценкой.
        """

        best_by_conditions: dict[
            tuple[tuple[str, str], ...],
            RuleEvaluation,
        ] = {}

        for evaluation in evaluations:
            condition_key = tuple(
                sorted(
                    (
                        name,
                        repr(value),
                    )
                    for name, value
                    in evaluation.conditions.items()
                )
            )

            current_best = best_by_conditions.get(
                condition_key
            )

            if current_best is None:
                best_by_conditions[
                    condition_key
                ] = evaluation

                continue

            if (
                self._evaluation_score(evaluation)
                > self._evaluation_score(current_best)
            ):
                best_by_conditions[
                    condition_key
                ] = evaluation

        return list(best_by_conditions.values())

    @staticmethod
    def _conditions_match(
        conditions: dict[str, Any],
        predicates: dict[str, Any],
    ) -> bool:
        return all(
            predicates.get(name) == expected_value
            for name, expected_value
            in conditions.items()
        )

    @staticmethod
    def _evaluation_score(
        evaluation: RuleEvaluation,
    ) -> tuple[float, float, float, int]:
        advantage = (
            evaluation.return_advantage
            if evaluation.return_advantage is not None
            else 0.0
        )

        return (
            evaluation.episode_success_rate,
            advantage,
            evaluation.average_return_to_go,
            evaluation.action_support,
        )