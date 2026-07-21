from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# КАНДИДАТ НА СИМВОЛИЧЕСКОЕ ПРАВИЛО
# ============================================================


@dataclass(frozen=True)
class NeuralRuleCandidate:
    """
    Обобщённое правило, найденное нейросетью.

    conditions:
        Условия правила в формате:
        {
            "has_key": True,
            "near_door": True,
            "door_locked": True,
        }

    action:
        Действие, которое рекомендуется выполнить.

    confidence:
        Средняя уверенность нейросети для этого действия.

    support:
        Количество опытов, поддерживающих правило.

    success_rate:
        Доля успешных опытов среди поддерживающих правило.
    """

    conditions: dict[str, Any]
    action: int
    confidence: float
    support: int
    success_rate: float

    @property
    def specificity(self) -> int:
        return len(self.conditions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conditions": dict(self.conditions),
            "action": self.action,
            "confidence": self.confidence,
            "support": self.support,
            "success_rate": self.success_rate,
            "specificity": self.specificity,
        }


# ============================================================
# ОБУЧАЮЩИЙ ПРИМЕР
# ============================================================


@dataclass(frozen=True)
class PatternExample:
    predicates: dict[str, Any]
    action: int
    reward: float
    success: bool


# ============================================================
# КОДИРОВЩИК ПРЕДИКАТОВ
# ============================================================


class PredicateVectorizer:
    """
    Преобразует словарь предикатов в бинарный вектор.

    Каждый признак представляется токеном:

        has_key=True
        door_locked=False
        key_direction=left

    Это позволяет работать не только с bool, но и с
    целыми числами, строками и None.
    """

    def __init__(self) -> None:
        self._tokens: tuple[str, ...] = ()
        self._token_to_index: dict[str, int] = {}

    @property
    def is_fitted(self) -> bool:
        return bool(self._tokens)

    @property
    def feature_count(self) -> int:
        return len(self._tokens)

    @property
    def tokens(self) -> tuple[str, ...]:
        return self._tokens

    def fit(
        self,
        predicate_sets: Iterable[dict[str, Any]],
    ) -> None:
        tokens: set[str] = set()

        for predicates in predicate_sets:
            self._validate_predicates(predicates)

            for name, value in predicates.items():
                tokens.add(
                    self.make_token(name, value)
                )

        if not tokens:
            raise ValueError(
                "cannot fit vectorizer on empty predicates"
            )

        self._tokens = tuple(sorted(tokens))

        self._token_to_index = {
            token: index
            for index, token in enumerate(self._tokens)
        }

    def transform(
        self,
        predicates: dict[str, Any],
    ) -> torch.Tensor:
        if not self.is_fitted:
            raise RuntimeError(
                "vectorizer must be fitted before transform"
            )

        self._validate_predicates(predicates)

        vector = torch.zeros(
            self.feature_count,
            dtype=torch.float32,
        )

        for name, value in predicates.items():
            token = self.make_token(name, value)

            index = self._token_to_index.get(token)

            if index is not None:
                vector[index] = 1.0

        return vector

    def transform_many(
        self,
        predicate_sets: Sequence[dict[str, Any]],
    ) -> torch.Tensor:
        if not predicate_sets:
            raise ValueError(
                "predicate_sets cannot be empty"
            )

        return torch.stack(
            [
                self.transform(predicates)
                for predicates in predicate_sets
            ]
        )

    def token_at(self, index: int) -> str:
        if not isinstance(index, int):
            raise TypeError("index must be int")

        return self._tokens[index]

    @staticmethod
    def make_token(
        name: str,
        value: Any,
    ) -> str:
        if not isinstance(name, str):
            raise TypeError(
                "predicate name must be str"
            )

        if not name.strip():
            raise ValueError(
                "predicate name cannot be empty"
            )

        return f"{name}={repr(value)}"

    @staticmethod
    def parse_token(
        token: str,
    ) -> tuple[str, Any]:
        """
        Преобразует токен обратно в условие.

        Поддерживает наиболее распространённые значения
        предикатов проекта.
        """

        if "=" not in token:
            raise ValueError(
                f"invalid predicate token: {token}"
            )

        name, raw_value = token.split("=", 1)

        if raw_value == "True":
            value: Any = True
        elif raw_value == "False":
            value = False
        elif raw_value == "None":
            value = None
        elif (
            len(raw_value) >= 2
            and raw_value[0] == raw_value[-1]
            and raw_value[0] in {"'", '"'}
        ):
            value = raw_value[1:-1]
        else:
            try:
                value = int(raw_value)
            except ValueError:
                try:
                    value = float(raw_value)
                except ValueError:
                    value = raw_value

        return name, value

    @staticmethod
    def _validate_predicates(
        predicates: dict[str, Any],
    ) -> None:
        if not isinstance(predicates, dict):
            raise TypeError(
                "predicates must be dict"
            )

        for name in predicates:
            if not isinstance(name, str):
                raise TypeError(
                    "predicate names must be strings"
                )

            if not name.strip():
                raise ValueError(
                    "predicate names cannot be empty"
                )


# ============================================================
# НЕЙРОСЕТЬ ПОИСКА ЗАКОНОМЕРНОСТЕЙ
# ============================================================


class PatternNetwork(nn.Module):

    def __init__(
        self,
        input_size: int,
        action_count: int,
        hidden_size: int = 64,
    ) -> None:
        super().__init__()

        if input_size <= 0:
            raise ValueError(
                "input_size must be greater than zero"
            )

        if action_count <= 1:
            raise ValueError(
                "action_count must be greater than one"
            )

        if hidden_size <= 0:
            raise ValueError(
                "hidden_size must be greater than zero"
            )

        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_count),
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        return self.layers(inputs)


# ============================================================
# NEURAL PATTERN MINER
# ============================================================


class NeuralPatternMiner:
    """
    Ищет повторяющиеся закономерности в накопленном опыте.

    Нейросеть обучается предсказывать полезное действие
    по предикатам состояния.

    После обучения miner:

    1. выбирает опыты, в которых сеть уверена;
    2. группирует их по действию;
    3. находит часто повторяющиеся предикаты;
    4. создаёт читаемые кандидаты на правила.
    """

    def __init__(
        self,
        actions: Sequence[int],
        hidden_size: int = 64,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        if not actions:
            raise ValueError(
                "actions cannot be empty"
            )

        if any(
            not isinstance(action, int)
            for action in actions
        ):
            raise TypeError(
                "all actions must be integers"
            )

        if len(set(actions)) != len(actions):
            raise ValueError(
                "actions must be unique"
            )

        if learning_rate <= 0:
            raise ValueError(
                "learning_rate must be greater than zero"
            )

        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero"
            )

        self.actions = tuple(actions)
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.seed = seed
        self.device = torch.device(device)

        self.vectorizer = PredicateVectorizer()

        self._action_to_index = {
            action: index
            for index, action in enumerate(self.actions)
        }

        self._index_to_action = {
            index: action
            for index, action in enumerate(self.actions)
        }

        self.network: PatternNetwork | None = None
        self.optimizer: torch.optim.Optimizer | None = None

        self.loss_history: list[float] = []

        torch.manual_seed(seed)

    @property
    def is_trained(self) -> bool:
        return (
            self.network is not None
            and bool(self.loss_history)
        )

    def fit(
        self,
        examples: Sequence[PatternExample],
        epochs: int = 100,
    ) -> list[float]:
        if not examples:
            raise ValueError(
                "examples cannot be empty"
            )

        if epochs <= 0:
            raise ValueError(
                "epochs must be greater than zero"
            )

        self._validate_examples(examples)

        # Для обучения правилам сначала используем только
        # положительный или успешный опыт.
        useful_examples = [
            example
            for example in examples
            if example.reward > 0 or example.success
        ]

        if not useful_examples:
            raise ValueError(
                "no positive or successful examples"
            )

        self.vectorizer.fit(
            [
                example.predicates
                for example in useful_examples
            ]
        )

        inputs = self.vectorizer.transform_many(
            [
                example.predicates
                for example in useful_examples
            ]
        )

        targets = torch.tensor(
            [
                self._action_to_index[
                    example.action
                ]
                for example in useful_examples
            ],
            dtype=torch.long,
        )

        dataset = TensorDataset(
            inputs,
            targets,
        )

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        loader = DataLoader(
            dataset,
            batch_size=min(
                self.batch_size,
                len(dataset),
            ),
            shuffle=True,
            generator=generator,
        )

        self.network = PatternNetwork(
            input_size=self.vectorizer.feature_count,
            action_count=len(self.actions),
            hidden_size=self.hidden_size,
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate,
        )

        loss_function = nn.CrossEntropyLoss()

        self.loss_history = []

        self.network.train()

        for _ in range(epochs):
            epoch_loss = 0.0
            batch_count = 0

            for batch_inputs, batch_targets in loader:
                batch_inputs = batch_inputs.to(
                    self.device
                )
                batch_targets = batch_targets.to(
                    self.device
                )

                self.optimizer.zero_grad()

                logits = self.network(batch_inputs)

                loss = loss_function(
                    logits,
                    batch_targets,
                )

                loss.backward()
                self.optimizer.step()

                epoch_loss += float(loss.item())
                batch_count += 1

            self.loss_history.append(
                epoch_loss / batch_count
            )

        return list(self.loss_history)

    def predict(
        self,
        predicates: dict[str, Any],
    ) -> tuple[int, float]:
        network = self._require_network()

        inputs = self.vectorizer.transform(
            predicates
        ).unsqueeze(0).to(self.device)

        network.eval()

        with torch.no_grad():
            logits = network(inputs)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )

            confidence_tensor, index_tensor = (
                probabilities.max(dim=1)
            )

        action_index = int(index_tensor.item())
        confidence = float(
            confidence_tensor.item()
        )

        action = self._index_to_action[
            action_index
        ]

        return action, confidence

    def extract_candidates(
        self,
        examples: Sequence[PatternExample],
        minimum_confidence: float = 0.75,
        minimum_support: int = 3,
        predicate_frequency: float = 0.8,
        maximum_conditions: int = 5,
    ) -> list[NeuralRuleCandidate]:
        """
        Извлекает символические кандидаты из обученной сети.

        predicate_frequency=0.8 означает:
        условие должно встречаться как минимум в 80%
        уверенных примеров одного действия.
        """

        self._require_network()

        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError(
                "minimum_confidence must be in [0, 1]"
            )

        if minimum_support <= 0:
            raise ValueError(
                "minimum_support must be greater than zero"
            )

        if not 0.0 < predicate_frequency <= 1.0:
            raise ValueError(
                "predicate_frequency must be in (0, 1]"
            )

        if maximum_conditions <= 0:
            raise ValueError(
                "maximum_conditions must be greater than zero"
            )

        self._validate_examples(examples)

        grouped_examples: dict[
            int,
            list[tuple[PatternExample, float]],
        ] = defaultdict(list)

        for example in examples:
            predicted_action, confidence = (
                self.predict(example.predicates)
            )

            # Берём только случаи, где сеть уверена
            # и её вывод соответствует выполненному действию.
            if (
                predicted_action == example.action
                and confidence >= minimum_confidence
            ):
                grouped_examples[
                    predicted_action
                ].append(
                    (example, confidence)
                )

        candidates: list[NeuralRuleCandidate] = []

        for action, group in grouped_examples.items():
            if len(group) < minimum_support:
                continue

            token_counter: Counter[str] = Counter()

            for example, _ in group:
                unique_tokens = {
                    self.vectorizer.make_token(
                        name,
                        value,
                    )
                    for name, value
                    in example.predicates.items()
                }

                token_counter.update(unique_tokens)

            required_count = (
                len(group) * predicate_frequency
            )

            frequent_tokens = [
                (token, count)
                for token, count
                in token_counter.items()
                if count >= required_count
            ]

            frequent_tokens.sort(
                key=lambda item: (
                    -item[1],
                    item[0],
                )
            )

            selected_tokens = frequent_tokens[
                :maximum_conditions
            ]

            if not selected_tokens:
                continue

            conditions: dict[str, Any] = {}

            for token, _ in selected_tokens:
                name, value = (
                    self.vectorizer.parse_token(token)
                )

                conditions[name] = value

            confidences = [
                confidence
                for _, confidence in group
            ]

            successful_count = sum(
                example.success
                for example, _ in group
            )

            candidate = NeuralRuleCandidate(
                conditions=conditions,
                action=action,
                confidence=(
                    sum(confidences)
                    / len(confidences)
                ),
                support=len(group),
                success_rate=(
                    successful_count
                    / len(group)
                ),
            )

            candidates.append(candidate)

        candidates.sort(
            key=lambda candidate: (
                -candidate.confidence,
                -candidate.success_rate,
                -candidate.support,
                -candidate.specificity,
            )
        )

        return candidates

    def feature_importance(
        self,
        predicates: dict[str, Any],
        action: int | None = None,
    ) -> dict[str, float]:
        """
        Оценивает важность каждого присутствующего предиката
        методом маскирования.

        1. Считаем вероятность действия с полным состоянием.
        2. По одному отключаем каждый признак.
        3. Измеряем падение вероятности.

        Чем сильнее упала вероятность, тем важнее признак.
        """

        network = self._require_network()

        if action is None:
            action, _ = self.predict(predicates)

        if action not in self._action_to_index:
            raise ValueError(
                f"unknown action: {action}"
            )

        vector = self.vectorizer.transform(
            predicates
        ).to(self.device)

        action_index = self._action_to_index[action]

        network.eval()

        with torch.no_grad():
            original_logits = network(
                vector.unsqueeze(0)
            )

            original_probability = float(
                torch.softmax(
                    original_logits,
                    dim=1,
                )[0, action_index].item()
            )

        importances: dict[str, float] = {}

        active_indices = torch.nonzero(
            vector > 0,
            as_tuple=False,
        ).flatten()

        for index_tensor in active_indices:
            index = int(index_tensor.item())

            masked_vector = vector.clone()
            masked_vector[index] = 0.0

            with torch.no_grad():
                masked_logits = network(
                    masked_vector.unsqueeze(0)
                )

                masked_probability = float(
                    torch.softmax(
                        masked_logits,
                        dim=1,
                    )[0, action_index].item()
                )

            importance = max(
                0.0,
                original_probability
                - masked_probability,
            )

            token = self.vectorizer.token_at(index)

            importances[token] = importance

        return dict(
            sorted(
                importances.items(),
                key=lambda item: -item[1],
            )
        )

    def _require_network(
        self,
    ) -> PatternNetwork:
        if self.network is None:
            raise RuntimeError(
                "miner must be fitted before use"
            )

        return self.network

    def _validate_examples(
        self,
        examples: Sequence[PatternExample],
    ) -> None:
        for example in examples:
            if not isinstance(
                example,
                PatternExample,
            ):
                raise TypeError(
                    "all examples must be PatternExample"
                )

            if example.action not in self._action_to_index:
                raise ValueError(
                    f"unknown action: {example.action}"
                )

            if not isinstance(
                example.predicates,
                dict,
            ):
                raise TypeError(
                    "example predicates must be dict"
                )