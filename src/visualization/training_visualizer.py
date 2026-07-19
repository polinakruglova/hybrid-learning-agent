from __future__ import annotations

import csv

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import matplotlib

# Позволяет сохранять графики без открытия отдельного окна.
matplotlib.use("Agg")

import matplotlib.pyplot as plt


class TrainingVisualizer:
    """
    Строит графики обучения по файлам training_history.csv.

    Поддерживает:

    1. Визуализацию одного запуска.
    2. Сравнение нескольких запусков.
    3. Сглаживание значений скользящим средним.
    """

    REQUIRED_COLUMNS = {
        "episode",
        "total_reward",
        "steps",
        "success",
        "epsilon",
        "rule_count",
        "q_state_count",
    }

    def __init__(
        self,
        output_directory: str | Path = "outputs/plots",
        rolling_window: int = 50,
    ) -> None:
        if rolling_window <= 0:
            raise ValueError(
                "rolling_window должен быть положительным."
            )

        self._output_directory = Path(output_directory)
        self._rolling_window = rolling_window

    @property
    def output_directory(self) -> Path:
        return self._output_directory

    @property
    def rolling_window(self) -> int:
        return self._rolling_window

    # ========================================================
    # ЗАГРУЗКА ДАННЫХ
    # ========================================================

    def load_history(
        self,
        csv_path: str | Path,
    ) -> list[dict[str, Any]]:
        """
        Загружает историю обучения из CSV.
        """

        path = Path(csv_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Файл истории не найден: {path}"
            )

        if not path.is_file():
            raise ValueError(
                f"Путь не является файлом: {path}"
            )

        with path.open(
            mode="r",
            encoding="utf-8",
            newline="",
        ) as file:
            reader = csv.DictReader(file)

            if reader.fieldnames is None:
                raise ValueError(
                    "CSV-файл не содержит заголовков."
                )

            missing_columns = (
                self.REQUIRED_COLUMNS
                - set(reader.fieldnames)
            )

            if missing_columns:
                missing_text = ", ".join(
                    sorted(missing_columns)
                )

                raise ValueError(
                    "В CSV отсутствуют обязательные столбцы: "
                    f"{missing_text}"
                )

            rows = [
                self._convert_row(row)
                for row in reader
            ]

        if not rows:
            raise ValueError(
                "История обучения не содержит эпизодов."
            )

        return rows

    @staticmethod
    def _convert_row(
        row: Mapping[str, str],
    ) -> dict[str, Any]:
        return {
            "episode": int(row["episode"]),
            "total_reward": float(
                row["total_reward"]
            ),
            "steps": int(row["steps"]),
            "success": (
                row["success"].strip().lower()
                == "true"
            ),
            "final_event": row.get(
                "final_event",
                "",
            ),
            "epsilon": float(row["epsilon"]),
            "rule_count": int(row["rule_count"]),
            "q_state_count": int(
                row["q_state_count"]
            ),
        }

    # ========================================================
    # ВИЗУАЛИЗАЦИЯ ОДНОГО ЗАПУСКА
    # ========================================================

    def plot_all(
        self,
        csv_path: str | Path,
        experiment_name: str = "Hybrid Agent",
    ) -> dict[str, Path]:
        """
        Создаёт полный набор графиков одного запуска.
        """

        history = self.load_history(csv_path)

        self._output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return {
            "reward": self.plot_reward(
                history,
                experiment_name,
            ),
            "success_rate": self.plot_success_rate(
                history,
                experiment_name,
            ),
            "episode_steps": self.plot_episode_steps(
                history,
                experiment_name,
            ),
            "epsilon": self.plot_epsilon(
                history,
                experiment_name,
            ),
            "rule_count": self.plot_rule_count(
                history,
                experiment_name,
            ),
            "q_state_count": self.plot_q_state_count(
                history,
                experiment_name,
            ),
        }

    def plot_reward(
        self,
        history: list[dict[str, Any]],
        experiment_name: str = "Hybrid Agent",
        filename: str = "reward.png",
    ) -> Path:
        episodes = self._values(
            history,
            "episode",
        )

        rewards = self._values(
            history,
            "total_reward",
        )

        smoothed_rewards = self.rolling_average(
            rewards
        )

        path = self._prepare_path(filename)

        plt.figure(figsize=(10, 6))

        plt.plot(
            episodes,
            rewards,
            alpha=0.25,
            label="Награда эпизода",
        )

        plt.plot(
            episodes,
            smoothed_rewards,
            linewidth=2,
            label=(
                "Скользящее среднее "
                f"({self._rolling_window})"
            ),
        )

        plt.title(
            f"Динамика награды — {experiment_name}"
        )
        plt.xlabel("Эпизод")
        plt.ylabel("Награда")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()

        return path

    def plot_success_rate(
        self,
        history: list[dict[str, Any]],
        experiment_name: str = "Hybrid Agent",
        filename: str = "success_rate.png",
    ) -> Path:
        episodes = self._values(
            history,
            "episode",
        )

        successes = [
            1.0 if row["success"] else 0.0
            for row in history
        ]

        success_rate = self.rolling_average(
            successes
        )

        success_percent = [
            value * 100.0
            for value in success_rate
        ]

        path = self._prepare_path(filename)

        plt.figure(figsize=(10, 6))

        plt.plot(
            episodes,
            success_percent,
            linewidth=2,
        )

        plt.title(
            "Скользящая доля успешных эпизодов — "
            f"{experiment_name}"
        )
        plt.xlabel("Эпизод")
        plt.ylabel("Успешность, %")
        plt.ylim(-2, 102)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()

        return path

    def plot_episode_steps(
        self,
        history: list[dict[str, Any]],
        experiment_name: str = "Hybrid Agent",
        filename: str = "episode_steps.png",
    ) -> Path:
        episodes = self._values(
            history,
            "episode",
        )

        steps = self._values(
            history,
            "steps",
        )

        smoothed_steps = self.rolling_average(
            steps
        )

        path = self._prepare_path(filename)

        plt.figure(figsize=(10, 6))

        plt.plot(
            episodes,
            steps,
            alpha=0.25,
            label="Шаги эпизода",
        )

        plt.plot(
            episodes,
            smoothed_steps,
            linewidth=2,
            label=(
                "Скользящее среднее "
                f"({self._rolling_window})"
            ),
        )

        plt.title(
            f"Длина эпизода — {experiment_name}"
        )
        plt.xlabel("Эпизод")
        plt.ylabel("Количество шагов")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()

        return path

    def plot_epsilon(
        self,
        history: list[dict[str, Any]],
        experiment_name: str = "Hybrid Agent",
        filename: str = "epsilon.png",
    ) -> Path:
        episodes = self._values(
            history,
            "episode",
        )

        epsilon = self._values(
            history,
            "epsilon",
        )

        path = self._prepare_path(filename)

        plt.figure(figsize=(10, 6))

        plt.plot(
            episodes,
            epsilon,
            linewidth=2,
        )

        plt.title(
            f"Изменение epsilon — {experiment_name}"
        )
        plt.xlabel("Эпизод")
        plt.ylabel("Epsilon")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()

        return path

    def plot_rule_count(
        self,
        history: list[dict[str, Any]],
        experiment_name: str = "Hybrid Agent",
        filename: str = "rule_count.png",
    ) -> Path:
        episodes = self._values(
            history,
            "episode",
        )

        rule_count = self._values(
            history,
            "rule_count",
        )

        path = self._prepare_path(filename)

        plt.figure(figsize=(10, 6))

        plt.plot(
            episodes,
            rule_count,
            linewidth=2,
        )

        plt.title(
            f"Количество правил — {experiment_name}"
        )
        plt.xlabel("Эпизод")
        plt.ylabel("Правила")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()

        return path

    def plot_q_state_count(
        self,
        history: list[dict[str, Any]],
        experiment_name: str = "Hybrid Agent",
        filename: str = "q_state_count.png",
    ) -> Path:
        episodes = self._values(
            history,
            "episode",
        )

        q_state_count = self._values(
            history,
            "q_state_count",
        )

        path = self._prepare_path(filename)

        plt.figure(figsize=(10, 6))

        plt.plot(
            episodes,
            q_state_count,
            linewidth=2,
        )

        plt.title(
            "Количество состояний Q-таблицы — "
            f"{experiment_name}"
        )
        plt.xlabel("Эпизод")
        plt.ylabel("Состояния Q-таблицы")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()

        return path

    # ========================================================
    # СРАВНЕНИЕ НЕСКОЛЬКИХ ЗАПУСКОВ
    # ========================================================

    def compare_experiments(
        self,
        experiments: Mapping[str, str | Path],
    ) -> dict[str, Path]:
        """
        Сравнивает несколько экспериментов.

        Пример:

        {
            "Hybrid Agent": "outputs/hybrid/training_history.csv",
            "Q Agent": "outputs/q_agent/training_history.csv",
            "Rule Agent": "outputs/rule_agent/training_history.csv",
        }
        """

        if not experiments:
            raise ValueError(
                "Нужно передать хотя бы один эксперимент."
            )

        loaded = {
            name: self.load_history(path)
            for name, path in experiments.items()
        }

        comparison_directory = (
            self._output_directory / "comparison"
        )

        comparison_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return {
            "reward": self._compare_metric(
                experiments=loaded,
                metric="total_reward",
                title="Сравнение средней награды",
                ylabel="Награда",
                output_path=(
                    comparison_directory
                    / "reward_comparison.png"
                ),
            ),
            "success_rate": (
                self._compare_success_rate(
                    experiments=loaded,
                    output_path=(
                        comparison_directory
                        / "success_rate_comparison.png"
                    ),
                )
            ),
            "episode_steps": self._compare_metric(
                experiments=loaded,
                metric="steps",
                title="Сравнение длины эпизода",
                ylabel="Количество шагов",
                output_path=(
                    comparison_directory
                    / "steps_comparison.png"
                ),
            ),
            "rule_count": self._compare_metric(
                experiments=loaded,
                metric="rule_count",
                title="Сравнение количества правил",
                ylabel="Правила",
                output_path=(
                    comparison_directory
                    / "rules_comparison.png"
                ),
            ),
            "q_state_count": self._compare_metric(
                experiments=loaded,
                metric="q_state_count",
                title=(
                    "Сравнение размера Q-таблицы"
                ),
                ylabel="Состояния Q-таблицы",
                output_path=(
                    comparison_directory
                    / "q_states_comparison.png"
                ),
            ),
        }

    def _compare_metric(
        self,
        experiments: Mapping[
            str,
            list[dict[str, Any]],
        ],
        metric: str,
        title: str,
        ylabel: str,
        output_path: Path,
    ) -> Path:
        plt.figure(figsize=(11, 7))

        for name, history in experiments.items():
            episodes = self._values(
                history,
                "episode",
            )

            values = self._values(
                history,
                metric,
            )

            smoothed = self.rolling_average(
                values
            )

            plt.plot(
                episodes,
                smoothed,
                linewidth=2,
                label=name,
            )

        plt.title(title)
        plt.xlabel("Эпизод")
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()

        return output_path

    def _compare_success_rate(
        self,
        experiments: Mapping[
            str,
            list[dict[str, Any]],
        ],
        output_path: Path,
    ) -> Path:
        plt.figure(figsize=(11, 7))

        for name, history in experiments.items():
            episodes = self._values(
                history,
                "episode",
            )

            successes = [
                1.0 if row["success"] else 0.0
                for row in history
            ]

            success_rate = self.rolling_average(
                successes
            )

            success_percent = [
                value * 100.0
                for value in success_rate
            ]

            plt.plot(
                episodes,
                success_percent,
                linewidth=2,
                label=name,
            )

        plt.title(
            "Сравнение успешности агентов"
        )
        plt.xlabel("Эпизод")
        plt.ylabel("Успешность, %")
        plt.ylim(-2, 102)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=160)
        plt.close()

        return output_path

    # ========================================================
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ========================================================

    def rolling_average(
        self,
        values: Iterable[float | int],
    ) -> list[float]:
        """
        Вычисляет скользящее среднее.

        Для первых эпизодов использует доступное
        количество значений.
        """

        value_list = [
            float(value)
            for value in values
        ]

        if not value_list:
            return []

        averages: list[float] = []
        running_sum = 0.0

        for index, value in enumerate(value_list):
            running_sum += value

            if index >= self._rolling_window:
                running_sum -= value_list[
                    index - self._rolling_window
                ]

            current_window = min(
                index + 1,
                self._rolling_window,
            )

            averages.append(
                running_sum / current_window
            )

        return averages

    def _prepare_path(
        self,
        filename: str,
    ) -> Path:
        if not isinstance(filename, str):
            raise TypeError(
                "filename должен иметь тип str."
            )

        if not filename.strip():
            raise ValueError(
                "filename не может быть пустым."
            )

        self._output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return self._output_directory / filename

    @staticmethod
    def _values(
        history: list[dict[str, Any]],
        key: str,
    ) -> list[Any]:
        return [
            row[key]
            for row in history
        ]