from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from src.memory import Experience, ExperienceBuffer
from src.predicates import PredicateGenerator
from src.rules.neural_pattern_miner import (
    NeuralPatternMiner,
    NeuralRuleCandidate,
    PatternExample,
)
from src.rules.rule import Rule
from src.rules.store import RuleStore


@dataclass(slots=True)
class NeuralRuleTrainingResult:
    """
    Результат одного запуска нейросетевой генерации правил.
    """

    examples_count: int
    candidates_count: int
    created_count: int
    updated_count: int
    rejected_count: int
    losses: tuple[float, ...]

    @property
    def accepted_count(self) -> int:
        return self.created_count + self.updated_count

    @property
    def final_loss(self) -> float | None:
        if not self.losses:
            return None

        return self.losses[-1]

    def to_dict(self) -> dict[str, object]:
        return {
            "examples_count": self.examples_count,
            "candidates_count": self.candidates_count,
            "created_count": self.created_count,
            "updated_count": self.updated_count,
            "rejected_count": self.rejected_count,
            "accepted_count": self.accepted_count,
            "final_loss": self.final_loss,
            "losses": list(self.losses),
        }


class NeuralRuleTrainer:
    """
    Нейросетевой генератор символических правил.

    Выполняет цепочку:

        ExperienceBuffer
                ↓
        PredicateGenerator
                ↓
        PatternExample
                ↓
        NeuralPatternMiner
                ↓
        NeuralRuleCandidate
                ↓
        Rule
                ↓
        RuleStore

    Старый RuleMiner этому классу не нужен.
    """

    def __init__(
        self,
        experience_buffer: ExperienceBuffer,
        predicate_generator: PredicateGenerator,
        pattern_miner: NeuralPatternMiner,
        rule_store: RuleStore,
        *,
        minimum_confidence: float = 0.70,
        minimum_support: int = 2,
        minimum_success_rate: float = 0.50,
        predicate_frequency: float = 0.60,
        maximum_conditions: int = 5,
        positive_only: bool = True,
    ) -> None:
        if not isinstance(
            experience_buffer,
            ExperienceBuffer,
        ):
            raise TypeError(
                "experience_buffer должен быть ExperienceBuffer."
            )

        if not isinstance(
            predicate_generator,
            PredicateGenerator,
        ):
            raise TypeError(
                "predicate_generator должен быть "
                "PredicateGenerator."
            )

        if not isinstance(
            pattern_miner,
            NeuralPatternMiner,
        ):
            raise TypeError(
                "pattern_miner должен быть NeuralPatternMiner."
            )

        if not isinstance(rule_store, RuleStore):
            raise TypeError(
                "rule_store должен быть RuleStore."
            )

        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError(
                "minimum_confidence должен быть от 0 до 1."
            )

        if minimum_support <= 0:
            raise ValueError(
                "minimum_support должен быть положительным."
            )

        if not 0.0 <= minimum_success_rate <= 1.0:
            raise ValueError(
                "minimum_success_rate должен быть от 0 до 1."
            )

        if not 0.0 < predicate_frequency <= 1.0:
            raise ValueError(
                "predicate_frequency должен быть больше 0 "
                "и не превышать 1."
            )

        if maximum_conditions <= 0:
            raise ValueError(
                "maximum_conditions должен быть положительным."
            )

        if not isinstance(positive_only, bool):
            raise TypeError(
                "positive_only должен иметь тип bool."
            )

        self.experience_buffer = experience_buffer
        self.predicate_generator = predicate_generator
        self.pattern_miner = pattern_miner
        self.rule_store = rule_store

        self.minimum_confidence = float(
            minimum_confidence
        )
        self.minimum_support = int(
            minimum_support
        )
        self.minimum_success_rate = float(
            minimum_success_rate
        )
        self.predicate_frequency = float(
            predicate_frequency
        )
        self.maximum_conditions = int(
            maximum_conditions
        )
        self.positive_only = positive_only

        self.training_runs = 0
        self.total_examples = 0
        self.total_candidates = 0
        self.total_created = 0
        self.total_updated = 0
        self.total_rejected = 0

        self._last_result: NeuralRuleTrainingResult | None = None

    @property
    def last_result(
        self,
    ) -> NeuralRuleTrainingResult | None:
        return self._last_result

    def collect_experiences(
        self,
    ) -> list[Experience]:
        """
        Возвращает опыт, который будет использован
        для обучения NeuralPatternMiner.

        При positive_only=True используются только:
        - успешные переходы;
        - переходы с положительной наградой.

        Отрицательные случайные действия не превращаются
        в обучающие примеры правил.
        """
        experiences = list(
            self.experience_buffer.experiences
        )

        if not self.positive_only:
            return experiences

        return [
            experience
            for experience in experiences
            if (
                experience.success
                or experience.reward > 0.0
            )
        ]

    def experience_to_example(
        self,
        experience: Experience,
    ) -> PatternExample:
        """
        Преобразует один переход Experience
        в пример для нейронной сети.
        """
        if not isinstance(experience, Experience):
            raise TypeError(
                "experience должен быть Experience."
            )

        predicates = self.predicate_generator.generate(
            experience.state
        )

        return PatternExample(
            predicates=predicates,
            action=experience.action,
            reward=float(experience.reward),
            success=experience.success,
        )

    def collect_examples(
        self,
        experiences: Iterable[Experience] | None = None,
    ) -> list[PatternExample]:
        """
        Преобразует накопленный опыт в PatternExample.
        """
        if experiences is None:
            selected_experiences = (
                self.collect_experiences()
            )
        else:
            selected_experiences = list(experiences)

        return [
            self.experience_to_example(experience)
            for experience in selected_experiences
        ]

    def train_from_buffer(
        self,
        *,
        epochs: int = 100,
        minimum_examples: int = 10,
    ) -> NeuralRuleTrainingResult | None:
        """
        Обучает сеть на текущем содержимом ExperienceBuffer.

        Возвращает None, если в буфере пока недостаточно
        подходящих примеров.
        """
        if epochs <= 0:
            raise ValueError(
                "epochs должен быть положительным."
            )

        if minimum_examples <= 0:
            raise ValueError(
                "minimum_examples должен быть положительным."
            )

        examples = self.collect_examples()

        if len(examples) < minimum_examples:
            return None

        return self.train(
            examples=examples,
            epochs=epochs,
        )

    def train(
        self,
        examples: Sequence[PatternExample],
        *,
        epochs: int = 100,
    ) -> NeuralRuleTrainingResult:
        """
        Обучает NeuralPatternMiner и обновляет RuleStore.
        """
        if not examples:
            raise ValueError(
                "Для обучения нужен хотя бы один пример."
            )

        if epochs <= 0:
            raise ValueError(
                "epochs должен быть положительным."
            )

        losses = self.pattern_miner.fit(
            examples,
            epochs=epochs,
        )

        candidates = (
            self.pattern_miner.extract_candidates(
                examples,
                minimum_confidence=(
                    self.minimum_confidence
                ),
                minimum_support=self.minimum_support,
                predicate_frequency=(
                    self.predicate_frequency
                ),
                maximum_conditions=(
                    self.maximum_conditions
                ),
            )
        )

        result = self.update_rule_store(
            candidates=candidates,
            examples_count=len(examples),
            losses=losses,
        )

        self.training_runs += 1
        self.total_examples += result.examples_count
        self.total_candidates += result.candidates_count
        self.total_created += result.created_count
        self.total_updated += result.updated_count
        self.total_rejected += result.rejected_count

        self._last_result = result

        return result

    def update_rule_store(
        self,
        candidates: Iterable[NeuralRuleCandidate],
        *,
        examples_count: int = 0,
        losses: Sequence[float] = (),
    ) -> NeuralRuleTrainingResult:
        """
        Проверяет кандидатов и записывает подходящие
        правила в RuleStore.
        """
        candidates_list = list(candidates)

        created_count = 0
        updated_count = 0
        rejected_count = 0

        for candidate in candidates_list:
            if not self.accept_candidate(candidate):
                rejected_count += 1
                continue

            rule = self.candidate_to_rule(candidate)

            existing_rule = (
                self.rule_store.find_equivalent(rule)
            )

            if existing_rule is None:
                self.rule_store.add(rule)
                created_count += 1
            else:
                self._update_existing_rule(
                    existing_rule=existing_rule,
                    candidate=candidate,
                )
                updated_count += 1

        return NeuralRuleTrainingResult(
            examples_count=examples_count,
            candidates_count=len(candidates_list),
            created_count=created_count,
            updated_count=updated_count,
            rejected_count=rejected_count,
            losses=tuple(
                float(loss)
                for loss in losses
            ),
        )

    def accept_candidate(
        self,
        candidate: NeuralRuleCandidate,
    ) -> bool:
        """
        Проверяет качество найденного правила.
        """
        if not isinstance(
            candidate,
            NeuralRuleCandidate,
        ):
            raise TypeError(
                "candidate должен быть NeuralRuleCandidate."
            )

        if not candidate.conditions:
            return False

        if candidate.support < self.minimum_support:
            return False

        if (
            candidate.confidence
            < self.minimum_confidence
        ):
            return False

        if (
            candidate.success_rate
            < self.minimum_success_rate
        ):
            return False

        if (
            len(candidate.conditions)
            > self.maximum_conditions
        ):
            return False

        return True

    @staticmethod
    def candidate_to_rule(
        candidate: NeuralRuleCandidate,
    ) -> Rule:
        """
        Преобразует NeuralRuleCandidate в Rule.
        """
        support = max(
            0,
            int(candidate.support),
        )

        successes = round(
            support * candidate.success_rate
        )

        successes = min(
            support,
            max(0, successes),
        )

        return Rule(
            conditions=dict(candidate.conditions),
            action=int(candidate.action),
            support=support,
            successes=successes,
            total_reward=0.0,
            metadata={
                "source": "neural_pattern_miner",
                "neural_confidence": float(
                    candidate.confidence
                ),
                "mined_success_rate": float(
                    candidate.success_rate
                ),
                "mined_support": support,
            },
        )

    @staticmethod
    def _update_existing_rule(
        existing_rule: Rule,
        candidate: NeuralRuleCandidate,
    ) -> None:
        """
        Обновляет эквивалентное правило,
        не создавая дубликат.
        """
        candidate_support = max(
            0,
            int(candidate.support),
        )

        candidate_successes = round(
            candidate_support
            * candidate.success_rate
        )

        candidate_successes = min(
            candidate_support,
            max(0, candidate_successes),
        )

        existing_neural_confidence = float(
            existing_rule.metadata.get(
                "neural_confidence",
                0.0,
            )
        )

        candidate_is_stronger = (
            candidate_support
            > existing_rule.support
            or candidate.confidence
            > existing_neural_confidence
        )

        if candidate_is_stronger:
            existing_rule.support = candidate_support
            existing_rule.successes = (
                candidate_successes
            )

        existing_rule.metadata.update(
            {
                "source": "neural_pattern_miner",
                "neural_confidence": max(
                    existing_neural_confidence,
                    float(candidate.confidence),
                ),
                "mined_success_rate": float(
                    candidate.success_rate
                ),
                "mined_support": candidate_support,
            }
        )

    def statistics(self) -> dict[str, int | float | None]:
        """
        Возвращает статистику нейросетевой
        генерации правил.
        """
        final_loss: float | None = None

        if self._last_result is not None:
            final_loss = self._last_result.final_loss

        return {
            "training_runs": self.training_runs,
            "total_examples": self.total_examples,
            "total_candidates": self.total_candidates,
            "total_created": self.total_created,
            "total_updated": self.total_updated,
            "total_rejected": self.total_rejected,
            "rules_in_store": len(self.rule_store),
            "last_final_loss": final_loss,
        }