from __future__ import annotations

from typing import Any

import pytest

from src.environments import KeyDoorEnvironment
from src.training import (
    EpisodeResult,
    TrainingHistory,
    TrainingRunner,
)


class FakeController:
    """
    Минимальный контроллер для тестирования TrainingRunner.

    Контроллер всегда выбирает действие ожидания,
    поэтому эпизод заканчивается после достижения max_steps.
    """

    def __init__(self) -> None:
        self.learn_calls: list[dict[str, Any]] = []
        self.decay_calls: list[dict[str, float]] = []

        self._epsilon = 1.0
        self._rule_count = 0
        self._q_state_count = 0

    def choose_action(self, state):
        return KeyDoorEnvironment.ACTION_WAIT

    def learn(
        self,
        state,
        action,
        reward,
        next_state,
        done=False,
        success=False,
        source=None,
    ):
        self.learn_calls.append(
            {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done,
                "success": success,
                "source": source,
            }
        )

        self._q_state_count += 1

    def decay_epsilon(
        self,
        decay_rate: float,
        minimum_epsilon: float = 0.01,
    ) -> float:
        self.decay_calls.append(
            {
                "decay_rate": decay_rate,
                "minimum_epsilon": minimum_epsilon,
            }
        )

        self._epsilon = max(
            minimum_epsilon,
            self._epsilon * decay_rate,
        )

        return self._epsilon

    def statistics(self) -> dict[str, Any]:
        return {
            "epsilon": self._epsilon,
            "rule_count": self._rule_count,
            "q_state_count": self._q_state_count,
        }

    def clear_learning_data(self) -> None:
        self.learn_calls.clear()
        self.decay_calls.clear()

        self._epsilon = 1.0
        self._rule_count = 0
        self._q_state_count = 0


class FakeTrainingRunner(TrainingRunner):
    """
    Тестовая версия TrainingRunner.

    Обходит строгую проверку типа LearningController,
    чтобы в unit-тестах можно было использовать FakeController.
    """

    def __init__(
        self,
        environment: KeyDoorEnvironment,
        controller: FakeController,
        epsilon_decay: float | None = None,
        minimum_epsilon: float = 0.01,
    ) -> None:
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


def test_episode_result_to_dict() -> None:
    result = EpisodeResult(
        episode=1,
        total_reward=5.0,
        steps=10,
        success=True,
        final_event="goal_reached",
        epsilon=0.5,
        rule_count=4,
        q_state_count=12,
    )

    data = result.to_dict()

    assert data["episode"] == 1
    assert data["total_reward"] == 5.0
    assert data["steps"] == 10
    assert data["success"] is True
    assert data["final_event"] == "goal_reached"
    assert data["epsilon"] == 0.5
    assert data["rule_count"] == 4
    assert data["q_state_count"] == 12


def test_training_history_empty_values() -> None:
    history = TrainingHistory()

    assert len(history) == 0
    assert history.success_count == 0
    assert history.success_rate == 0.0
    assert history.total_reward == 0.0
    assert history.average_reward == 0.0
    assert history.average_steps == 0.0
    assert history.best_reward is None
    assert history.worst_reward is None


def test_training_history_statistics() -> None:
    history = TrainingHistory()

    history.add(
        EpisodeResult(
            episode=1,
            total_reward=2.0,
            steps=10,
            success=False,
            final_event="max_steps_reached",
            epsilon=0.9,
            rule_count=1,
            q_state_count=5,
        )
    )

    history.add(
        EpisodeResult(
            episode=2,
            total_reward=8.0,
            steps=6,
            success=True,
            final_event="goal_reached",
            epsilon=0.8,
            rule_count=2,
            q_state_count=8,
        )
    )

    assert len(history) == 2
    assert history.rewards == [2.0, 8.0]
    assert history.steps == [10, 6]
    assert history.successes == [False, True]

    assert history.success_count == 1
    assert history.success_rate == pytest.approx(0.5)

    assert history.total_reward == pytest.approx(10.0)
    assert history.average_reward == pytest.approx(5.0)
    assert history.average_steps == pytest.approx(8.0)

    assert history.best_reward == pytest.approx(8.0)
    assert history.worst_reward == pytest.approx(2.0)


def test_recent_success_rate() -> None:
    history = TrainingHistory()

    for episode in range(1, 6):
        history.add(
            EpisodeResult(
                episode=episode,
                total_reward=0.0,
                steps=1,
                success=episode >= 4,
                final_event="test",
                epsilon=0.0,
                rule_count=0,
                q_state_count=0,
            )
        )

    assert history.recent_success_rate(
        window=2
    ) == pytest.approx(1.0)

    assert history.recent_success_rate(
        window=5
    ) == pytest.approx(0.4)


def test_recent_average_reward() -> None:
    history = TrainingHistory()

    for episode, reward in enumerate(
        [1.0, 2.0, 3.0, 4.0],
        start=1,
    ):
        history.add(
            EpisodeResult(
                episode=episode,
                total_reward=reward,
                steps=1,
                success=False,
                final_event="test",
                epsilon=0.0,
                rule_count=0,
                q_state_count=0,
            )
        )

    assert history.recent_average_reward(
        window=2
    ) == pytest.approx(3.5)


def test_run_episode() -> None:
    environment = KeyDoorEnvironment(
        max_steps=3
    )

    controller = FakeController()

    runner = FakeTrainingRunner(
        environment=environment,
        controller=controller,
    )

    result = runner.run_episode(
        episode_number=1
    )

    assert isinstance(result, EpisodeResult)
    assert result.episode == 1
    assert result.steps == 3
    assert result.success is False
    assert result.final_event == (
        "max_steps_reached"
    )

    assert len(controller.learn_calls) == 3
    assert len(runner.history) == 1


def test_run_episode_decays_epsilon() -> None:
    environment = KeyDoorEnvironment(
        max_steps=1
    )

    controller = FakeController()

    runner = FakeTrainingRunner(
        environment=environment,
        controller=controller,
        epsilon_decay=0.5,
        minimum_epsilon=0.1,
    )

    result = runner.run_episode(
        episode_number=1
    )

    assert len(controller.decay_calls) == 1
    assert result.epsilon == pytest.approx(0.5)


def test_train_runs_requested_episodes() -> None:
    environment = KeyDoorEnvironment(
        max_steps=2
    )

    controller = FakeController()

    runner = FakeTrainingRunner(
        environment=environment,
        controller=controller,
    )

    history = runner.train(
        episodes=4,
        log_interval=None,
    )

    assert len(history) == 4

    assert [
        result.episode
        for result in history
    ] == [1, 2, 3, 4]

    assert len(controller.learn_calls) == 8


def test_train_continues_episode_numbers() -> None:
    environment = KeyDoorEnvironment(
        max_steps=1
    )

    controller = FakeController()

    runner = FakeTrainingRunner(
        environment=environment,
        controller=controller,
    )

    runner.train(
        episodes=2,
        log_interval=None,
    )

    runner.train(
        episodes=2,
        log_interval=None,
    )

    assert [
        result.episode
        for result in runner.history
    ] == [1, 2, 3, 4]


def test_reset_history_keeps_learning_data() -> None:
    environment = KeyDoorEnvironment(
        max_steps=2
    )

    controller = FakeController()

    runner = FakeTrainingRunner(
        environment=environment,
        controller=controller,
    )

    runner.train(
        episodes=1,
        log_interval=None,
    )

    assert len(runner.history) == 1
    assert len(controller.learn_calls) == 2

    runner.reset_history()

    assert len(runner.history) == 0
    assert len(controller.learn_calls) == 2


def test_reset_all_clears_everything() -> None:
    environment = KeyDoorEnvironment(
        max_steps=2
    )

    controller = FakeController()

    runner = FakeTrainingRunner(
        environment=environment,
        controller=controller,
    )

    runner.train(
        episodes=1,
        log_interval=None,
    )

    runner.reset_all()

    assert len(runner.history) == 0
    assert len(controller.learn_calls) == 0
    assert environment.step_count == 0
    assert environment.done is False


@pytest.mark.parametrize(
    "episodes",
    [0, -1],
)
def test_train_rejects_invalid_episode_count(
    episodes: int,
) -> None:
    environment = KeyDoorEnvironment()
    controller = FakeController()

    runner = FakeTrainingRunner(
        environment=environment,
        controller=controller,
    )

    with pytest.raises(ValueError):
        runner.train(
            episodes=episodes
        )


@pytest.mark.parametrize(
    "window",
    [0, -1],
)
def test_history_rejects_invalid_window(
    window: int,
) -> None:
    history = TrainingHistory()

    with pytest.raises(ValueError):
        history.recent_success_rate(
            window=window
        )

    with pytest.raises(ValueError):
        history.recent_average_reward(
            window=window
        )