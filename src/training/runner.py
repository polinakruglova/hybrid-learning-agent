from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.environments import KeyDoorEnvironment
from src.learning import LearningController


@dataclass(frozen=True)
class EpisodeResult:
    """
    Результат одного эпизода обучения.
    """

    episode: int
    total_reward: float
    steps: int
    success: bool
    final_event: str
    epsilon: float
    rule_count: int
    q_state_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "total_reward": self.total_reward,
            "steps": self.steps,
            "success": self.success,
            "final_event": self.final_event,
            "epsilon": self.epsilon,
            "rule_count": self.rule_count,
            "q_state_count": self.q_state_count,
        }


@dataclass
class TrainingHistory:
    """
    История нескольких эпизодов обучения.
    """

    episodes: list[EpisodeResult] = field(
        default_factory=list
    )

    def add(self, result: EpisodeResult) -> None:
        if not isinstance(result, EpisodeResult):
            raise TypeError(
                "result должен быть объектом EpisodeResult."
            )

        self.episodes.append(result)

    def clear(self) -> None:
        self.episodes.clear()

    def __len__(self) -> int:
        return len(self.episodes)

    def __iter__(self):
        return iter(self.episodes)

    @property
    def rewards(self) -> list[float]:
        return [
            episode.total_reward
            for episode in self.episodes
        ]

    @property
    def steps(self) -> list[int]:
        return [
            episode.steps
            for episode in self.episodes
        ]

    @property
    def successes(self) -> list[bool]:
        return [
            episode.success
            for episode in self.episodes
        ]

    @property
    def epsilons(self) -> list[float]:
        return [
            episode.epsilon
            for episode in self.episodes
        ]

    @property
    def rule_counts(self) -> list[int]:
        return [
            episode.rule_count
            for episode in self.episodes
        ]

    @property
    def q_state_counts(self) -> list[int]:
        return [
            episode.q_state_count
            for episode in self.episodes
        ]

    @property
    def success_count(self) -> int:
        return sum(self.successes)

    @property
    def success_rate(self) -> float:
        if not self.episodes:
            return 0.0

        return self.success_count / len(self.episodes)

    @property
    def total_reward(self) -> float:
        return sum(self.rewards)

    @property
    def average_reward(self) -> float:
        if not self.episodes:
            return 0.0

        return self.total_reward / len(self.episodes)

    @property
    def average_steps(self) -> float:
        if not self.episodes:
            return 0.0

        return sum(self.steps) / len(self.episodes)

    @property
    def best_reward(self) -> float | None:
        if not self.episodes:
            return None

        return max(self.rewards)

    @property
    def worst_reward(self) -> float | None:
        if not self.episodes:
            return None

        return min(self.rewards)

    def recent_success_rate(
        self,
        window: int = 100,
    ) -> float:
        """
        Доля успешных эпизодов в последнем окне.
        """
        if not isinstance(window, int):
            raise TypeError(
                "window должен иметь тип int."
            )

        if window <= 0:
            raise ValueError(
                "window должен быть положительным."
            )

        if not self.episodes:
            return 0.0

        recent = self.episodes[-window:]

        success_count = sum(
            episode.success
            for episode in recent
        )

        return success_count / len(recent)

    def recent_average_reward(
        self,
        window: int = 100,
    ) -> float:
        """
        Средняя награда в последних эпизодах.
        """
        if not isinstance(window, int):
            raise TypeError(
                "window должен иметь тип int."
            )

        if window <= 0:
            raise ValueError(
                "window должен быть положительным."
            )

        if not self.episodes:
            return 0.0

        recent = self.episodes[-window:]

        return (
            sum(
                episode.total_reward
                for episode in recent
            )
            / len(recent)
        )

    def to_dicts(self) -> list[dict[str, Any]]:
        return [
            episode.to_dict()
            for episode in self.episodes
        ]

    def summary(self) -> dict[str, Any]:
        return {
            "episode_count": len(self),
            "success_count": self.success_count,
            "success_rate": self.success_rate,
            "total_reward": self.total_reward,
            "average_reward": self.average_reward,
            "average_steps": self.average_steps,
            "best_reward": self.best_reward,
            "worst_reward": self.worst_reward,
        }


class TrainingRunner:
    """
    Запускает обучение гибридного агента в среде.

    На каждом шаге:

    1. Агент выбирает действие.
    2. Среда выполняет действие.
    3. Контроллер сохраняет опыт.
    4. Обновляется Q-таблица.
    5. RuleMiner может создать новое правило.
    """

    def __init__(
        self,
        environment: KeyDoorEnvironment,
        controller: LearningController,
        epsilon_decay: float | None = None,
        minimum_epsilon: float = 0.01,
    ) -> None:
        if not isinstance(
            environment,
            KeyDoorEnvironment,
        ):
            raise TypeError(
                "environment должен быть объектом "
                "KeyDoorEnvironment."
            )

        if not isinstance(
            controller,
            LearningController,
        ):
            raise TypeError(
                "controller должен быть объектом "
                "LearningController."
            )

        if (
            epsilon_decay is not None
            and not isinstance(
                epsilon_decay,
                (int, float),
            )
        ):
            raise TypeError(
                "epsilon_decay должен быть числом "
                "или None."
            )

        if (
            epsilon_decay is not None
            and not 0.0 < epsilon_decay <= 1.0
        ):
            raise ValueError(
                "epsilon_decay должен находиться "
                "в диапазоне (0, 1]."
            )

        if not isinstance(
            minimum_epsilon,
            (int, float),
        ):
            raise TypeError(
                "minimum_epsilon должен быть числом."
            )

        if not 0.0 <= minimum_epsilon <= 1.0:
            raise ValueError(
                "minimum_epsilon должен находиться "
                "в диапазоне [0, 1]."
            )

        self._environment = environment
        self._controller = controller
        self._epsilon_decay = (
            float(epsilon_decay)
            if epsilon_decay is not None
            else None
        )
        self._minimum_epsilon = float(
            minimum_epsilon
        )

        self._history = TrainingHistory()

    @property
    def environment(
        self,
    ) -> KeyDoorEnvironment:
        return self._environment

    @property
    def controller(
        self,
    ) -> LearningController:
        return self._controller

    @property
    def history(self) -> TrainingHistory:
        return self._history

    @property
    def epsilon_decay(self) -> float | None:
        return self._epsilon_decay

    @property
    def minimum_epsilon(self) -> float:
        return self._minimum_epsilon

    def run_episode(
        self,
        episode_number: int,
        render: bool = False,
    ) -> EpisodeResult:
        """
        Запускает один полный эпизод.
        """
        if not isinstance(episode_number, int):
            raise TypeError(
                "episode_number должен иметь тип int."
            )

        if episode_number <= 0:
            raise ValueError(
                "episode_number должен быть положительным."
            )

        if not isinstance(render, bool):
            raise TypeError(
                "render должен иметь тип bool."
            )

        state = self._environment.reset()

        total_reward = 0.0
        final_event = "episode_started"

        if render:
            print(
                f"\nEpisode {episode_number}"
            )
            print(self._environment.render())

        while not self._environment.done:
            action = self._controller.choose_action(
                state
            )

            step_result = self._environment.step(
                action
            )

            self._controller.learn(
                state=state,
                action=action,
                reward=step_result.reward,
                next_state=step_result.state,
                done=step_result.done,
                success=step_result.success,
            )

            total_reward += step_result.reward
            final_event = step_result.info["event"]
            state = step_result.state

            if render:
                print(
                    "\n"
                    f"Step: "
                    f"{self._environment.step_count}"
                )
                print(
                    f"Action: "
                    f"{step_result.info['action_name']}"
                )
                print(
                    f"Reward: {step_result.reward}"
                )
                print(
                    f"Event: {final_event}"
                )
                print(self._environment.render())

        if self._epsilon_decay is not None:
            self._controller.decay_epsilon(
                decay_rate=self._epsilon_decay,
                minimum_epsilon=(
                    self._minimum_epsilon
                ),
            )

        statistics = (
            self._controller.statistics()
        )

        result = EpisodeResult(
            episode=episode_number,
            total_reward=float(total_reward),
            steps=self._environment.step_count,
            success=self._environment.success,
            final_event=final_event,
            epsilon=float(
                statistics["epsilon"]
            ),
            rule_count=int(
                statistics["rule_count"]
            ),
            q_state_count=int(
                statistics["q_state_count"]
            ),
        )

        self._history.add(result)

        return result

    def train(
        self,
        episodes: int,
        render: bool = False,
        log_interval: int | None = 100,
    ) -> TrainingHistory:
        """
        Запускает несколько эпизодов обучения.
        """
        if not isinstance(episodes, int):
            raise TypeError(
                "episodes должен иметь тип int."
            )

        if episodes <= 0:
            raise ValueError(
                "episodes должен быть положительным."
            )

        if not isinstance(render, bool):
            raise TypeError(
                "render должен иметь тип bool."
            )

        if (
            log_interval is not None
            and not isinstance(log_interval, int)
        ):
            raise TypeError(
                "log_interval должен иметь тип int "
                "или None."
            )

        if (
            log_interval is not None
            and log_interval <= 0
        ):
            raise ValueError(
                "log_interval должен быть "
                "положительным."
            )

        start_episode = len(self._history) + 1
        end_episode = (
            start_episode + episodes
        )

        for episode_number in range(
            start_episode,
            end_episode,
        ):
            result = self.run_episode(
                episode_number=episode_number,
                render=render,
            )

            if (
                log_interval is not None
                and (
                    episode_number % log_interval == 0
                    or episode_number
                    == end_episode - 1
                )
            ):
                self._print_progress(result)

        return self._history

    def evaluate(
        self,
        episodes: int,
        render: bool = False,
    ) -> TrainingHistory:
        """
        Временная базовая оценка агента.

        В текущей версии метод запускает эпизоды без
        epsilon decay, но Q-learning продолжает обновляться.

        Полностью независимую оценку без обучения добавим
        после введения режима inference/evaluation.
        """
        previous_decay = self._epsilon_decay
        self._epsilon_decay = None

        evaluation_history = TrainingHistory()

        try:
            for episode_number in range(
                1,
                episodes + 1,
            ):
                result = self.run_episode(
                    episode_number=episode_number,
                    render=render,
                )

                evaluation_history.add(result)
        finally:
            self._epsilon_decay = previous_decay

        return evaluation_history

    def reset_history(self) -> None:
        """
        Очищает историю запусков.

        Обученные Q-значения, правила и опыт
        при этом сохраняются.
        """
        self._history.clear()

    def reset_all(self) -> None:
        """
        Очищает историю и все данные обучения.
        """
        self._history.clear()
        self._controller.clear_learning_data()
        self._environment.reset()

    def _print_progress(
        self,
        result: EpisodeResult,
    ) -> None:
        recent_success_rate = (
            self._history.recent_success_rate(
                window=100
            )
        )

        recent_reward = (
            self._history.recent_average_reward(
                window=100
            )
        )

        print(
            f"Episode {result.episode:>5} | "
            f"Reward {result.total_reward:>8.3f} | "
            f"Steps {result.steps:>3} | "
            f"Success {result.success!s:<5} | "
            f"Recent success "
            f"{recent_success_rate:>6.1%} | "
            f"Recent reward "
            f"{recent_reward:>8.3f} | "
            f"Rules {result.rule_count:>4} | "
            f"Q states "
            f"{result.q_state_count:>4} | "
            f"Epsilon {result.epsilon:.4f}"
        )