from __future__ import annotations

import csv
import json

from pathlib import Path

import pytest

from src.persistence import TrainingExporter
from src.rl import QTablePolicy
from src.rules import RuleStore
from src.training import (
    EpisodeResult,
    TrainingHistory,
)


def create_history() -> TrainingHistory:
    history = TrainingHistory()

    history.add(
        EpisodeResult(
            episode=1,
            total_reward=-1.5,
            steps=20,
            success=False,
            final_event="max_steps_reached",
            epsilon=0.9,
            rule_count=0,
            q_state_count=5,
        )
    )

    history.add(
        EpisodeResult(
            episode=2,
            total_reward=12.5,
            steps=14,
            success=True,
            final_event="goal_reached",
            epsilon=0.8,
            rule_count=2,
            q_state_count=12,
        )
    )

    return history


def create_q_policy() -> QTablePolicy:
    return QTablePolicy(
        actions=[0, 1, 2],
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.5,
        seed=42,
    )


def test_exporter_creation(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "outputs"

    exporter = TrainingExporter(
        output_directory=output_directory
    )

    assert (
        exporter.output_directory
        == output_directory
    )


def test_export_history_csv(
    tmp_path: Path,
) -> None:
    exporter = TrainingExporter(tmp_path)
    history = create_history()

    path = exporter.export_history_csv(
        history
    )

    assert path.exists()
    assert path.name == "training_history.csv"

    with path.open(
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(
            csv.DictReader(file)
        )

    assert len(rows) == 2
    assert rows[0]["episode"] == "1"
    assert rows[0]["success"] == "False"
    assert rows[1]["episode"] == "2"
    assert rows[1]["success"] == "True"
    assert rows[1]["final_event"] == (
        "goal_reached"
    )


def test_export_history_json(
    tmp_path: Path,
) -> None:
    exporter = TrainingExporter(tmp_path)
    history = create_history()

    path = exporter.export_history_json(
        history
    )

    assert path.exists()

    with path.open(
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    assert len(data) == 2
    assert data[0]["episode"] == 1
    assert data[1]["success"] is True
    assert data[1]["total_reward"] == 12.5


def test_export_summary(
    tmp_path: Path,
) -> None:
    exporter = TrainingExporter(tmp_path)
    history = create_history()

    path = exporter.export_summary(
        history=history,
        controller_statistics={
            "total_steps": 34,
            "epsilon": 0.5,
        },
        buffer_statistics={
            "size": 34,
        },
        configuration={
            "episodes": 2,
            "seed": 42,
        },
    )

    assert path.exists()

    with path.open(
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    assert data["training"][
        "episode_count"
    ] == 2

    assert data["training"][
        "success_count"
    ] == 1

    assert data["controller"][
        "total_steps"
    ] == 34

    assert data["experience_buffer"][
        "size"
    ] == 34

    assert data["configuration"][
        "seed"
    ] == 42


def test_export_rules(
    tmp_path: Path,
) -> None:
    exporter = TrainingExporter(tmp_path)
    rule_store = RuleStore()

    path = exporter.export_rules(
        rule_store
    )

    assert path.exists()

    with path.open(
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    assert data == []


def test_export_q_table(
    tmp_path: Path,
) -> None:
    exporter = TrainingExporter(tmp_path)
    q_policy = create_q_policy()

    path = exporter.export_q_table(
        q_policy
    )

    assert path.exists()

    with path.open(
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    assert isinstance(data, dict)


def test_export_all(
    tmp_path: Path,
) -> None:
    exporter = TrainingExporter(
        tmp_path / "results"
    )

    history = create_history()
    rule_store = RuleStore()
    q_policy = create_q_policy()

    paths = exporter.export_all(
        history=history,
        rule_store=rule_store,
        q_policy=q_policy,
        controller_statistics={
            "total_steps": 34,
        },
        buffer_statistics={
            "size": 34,
        },
        configuration={
            "episodes": 2,
        },
    )

    assert set(paths) == {
        "history_csv",
        "history_json",
        "summary",
        "rules",
        "q_table",
    }

    for path in paths.values():
        assert path.exists()


def test_exporter_creates_nested_directory(
    tmp_path: Path,
) -> None:
    output_directory = (
        tmp_path
        / "experiment"
        / "run_001"
        / "outputs"
    )

    exporter = TrainingExporter(
        output_directory
    )

    path = exporter.export_history_csv(
        create_history()
    )

    assert output_directory.exists()
    assert path.exists()


def test_custom_filename(
    tmp_path: Path,
) -> None:
    exporter = TrainingExporter(tmp_path)

    path = exporter.export_history_json(
        history=create_history(),
        filename="episodes.json",
    )

    assert path.name == "episodes.json"
    assert path.exists()


@pytest.mark.parametrize(
    "filename",
    ["", "   "],
)
def test_rejects_empty_filename(
    tmp_path: Path,
    filename: str,
) -> None:
    exporter = TrainingExporter(tmp_path)

    with pytest.raises(ValueError):
        exporter.export_history_csv(
            history=create_history(),
            filename=filename,
        )


def test_rejects_invalid_history(
    tmp_path: Path,
) -> None:
    exporter = TrainingExporter(tmp_path)

    with pytest.raises(TypeError):
        exporter.export_history_csv(
            history=[],
        )


def test_rejects_invalid_rule_store(
    tmp_path: Path,
) -> None:
    exporter = TrainingExporter(tmp_path)

    with pytest.raises(TypeError):
        exporter.export_rules(
            rule_store=[],
        )


def test_rejects_invalid_q_policy(
    tmp_path: Path,
) -> None:
    exporter = TrainingExporter(tmp_path)

    with pytest.raises(TypeError):
        exporter.export_q_table(
            q_policy={},
        )