from __future__ import annotations

import csv

from pathlib import Path

import pytest

from src.visualization import TrainingVisualizer


def create_history_csv(
    path: Path,
    episodes: int = 20,
    reward_offset: float = 0.0,
) -> Path:
    fieldnames = [
        "episode",
        "total_reward",
        "steps",
        "success",
        "final_event",
        "epsilon",
        "rule_count",
        "q_state_count",
    ]

    with path.open(
        mode="w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for episode in range(1, episodes + 1):
            success = episode > episodes // 2

            writer.writerow(
                {
                    "episode": episode,
                    "total_reward": (
                        reward_offset + episode
                    ),
                    "steps": max(
                        5,
                        30 - episode,
                    ),
                    "success": success,
                    "final_event": (
                        "goal_reached"
                        if success
                        else "max_steps_reached"
                    ),
                    "epsilon": max(
                        0.05,
                        1.0 - episode * 0.04,
                    ),
                    "rule_count": (
                        2 if success else 0
                    ),
                    "q_state_count": (
                        episode * 3
                    ),
                }
            )

    return path


def test_visualizer_creation(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "plots"

    visualizer = TrainingVisualizer(
        output_directory=output_directory,
        rolling_window=10,
    )

    assert (
        visualizer.output_directory
        == output_directory
    )

    assert visualizer.rolling_window == 10


def test_rejects_invalid_rolling_window() -> None:
    with pytest.raises(ValueError):
        TrainingVisualizer(
            rolling_window=0
        )


def test_load_history(
    tmp_path: Path,
) -> None:
    csv_path = create_history_csv(
        tmp_path / "history.csv"
    )

    visualizer = TrainingVisualizer(
        tmp_path / "plots"
    )

    history = visualizer.load_history(
        csv_path
    )

    assert len(history) == 20
    assert history[0]["episode"] == 1
    assert isinstance(
        history[0]["total_reward"],
        float,
    )
    assert isinstance(
        history[0]["success"],
        bool,
    )


def test_missing_history_file(
    tmp_path: Path,
) -> None:
    visualizer = TrainingVisualizer(
        tmp_path / "plots"
    )

    with pytest.raises(FileNotFoundError):
        visualizer.load_history(
            tmp_path / "missing.csv"
        )


def test_rolling_average() -> None:
    visualizer = TrainingVisualizer(
        rolling_window=3
    )

    result = visualizer.rolling_average(
        [1, 2, 3, 4, 5]
    )

    assert result == pytest.approx(
        [
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
        ]
    )


def test_plot_all(
    tmp_path: Path,
) -> None:
    csv_path = create_history_csv(
        tmp_path / "history.csv"
    )

    visualizer = TrainingVisualizer(
        output_directory=(
            tmp_path / "plots"
        ),
        rolling_window=5,
    )

    paths = visualizer.plot_all(
        csv_path=csv_path,
        experiment_name="Test Agent",
    )

    assert set(paths) == {
        "reward",
        "success_rate",
        "episode_steps",
        "epsilon",
        "rule_count",
        "q_state_count",
    }

    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0


def test_compare_experiments(
    tmp_path: Path,
) -> None:
    hybrid_path = create_history_csv(
        tmp_path / "hybrid.csv",
        reward_offset=10.0,
    )

    q_agent_path = create_history_csv(
        tmp_path / "q_agent.csv",
        reward_offset=5.0,
    )

    visualizer = TrainingVisualizer(
        output_directory=(
            tmp_path / "plots"
        ),
        rolling_window=5,
    )

    paths = visualizer.compare_experiments(
        {
            "Hybrid Agent": hybrid_path,
            "Q Agent": q_agent_path,
        }
    )

    assert set(paths) == {
        "reward",
        "success_rate",
        "episode_steps",
        "rule_count",
        "q_state_count",
    }

    for path in paths.values():
        assert path.exists()
        assert path.stat().st_size > 0


def test_compare_requires_experiment(
    tmp_path: Path,
) -> None:
    visualizer = TrainingVisualizer(
        output_directory=tmp_path
    )

    with pytest.raises(ValueError):
        visualizer.compare_experiments({})