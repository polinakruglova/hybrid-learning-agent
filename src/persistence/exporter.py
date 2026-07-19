from __future__ import annotations

import csv
import json

from pathlib import Path
from typing import Any

from src.rl import QTablePolicy
from src.rules import RuleStore
from src.training import TrainingHistory


class TrainingExporter:
    """
    Сохраняет результаты обучения в файлы.

    Создаёт:

    - training_history.csv
    - training_history.json
    - summary.json
    - rules.json
    - q_table.json
    """

    def __init__(
        self,
        output_directory: str | Path = "outputs",
    ) -> None:
        self._output_directory = Path(
            output_directory
        )

    @property
    def output_directory(self) -> Path:
        return self._output_directory

    def export_all(
        self,
        history: TrainingHistory,
        rule_store: RuleStore,
        q_policy: QTablePolicy,
        controller_statistics: dict[str, Any] | None = None,
        buffer_statistics: dict[str, Any] | None = None,
        configuration: dict[str, Any] | None = None,
    ) -> dict[str, Path]:
        """
        Сохраняет все результаты обучения.

        Возвращает словарь с путями созданных файлов.
        """

        self._validate_history(history)
        self._validate_rule_store(rule_store)
        self._validate_q_policy(q_policy)

        self._output_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        paths = {
            "history_csv": self.export_history_csv(
                history
            ),
            "history_json": self.export_history_json(
                history
            ),
            "summary": self.export_summary(
                history=history,
                controller_statistics=(
                    controller_statistics
                ),
                buffer_statistics=(
                    buffer_statistics
                ),
                configuration=configuration,
            ),
            "rules": self.export_rules(
                rule_store
            ),
            "q_table": self.export_q_table(
                q_policy
            ),
        }

        return paths

    def export_history_csv(
        self,
        history: TrainingHistory,
        filename: str = "training_history.csv",
    ) -> Path:
        """
        Сохраняет историю эпизодов в CSV.
        """

        self._validate_history(history)

        path = self._prepare_path(filename)

        rows = history.to_dicts()

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

            for row in rows:
                writer.writerow(row)

        return path

    def export_history_json(
        self,
        history: TrainingHistory,
        filename: str = "training_history.json",
    ) -> Path:
        """
        Сохраняет историю эпизодов в JSON.
        """

        self._validate_history(history)

        path = self._prepare_path(filename)

        self._write_json(
            path=path,
            data=history.to_dicts(),
        )

        return path

    def export_summary(
        self,
        history: TrainingHistory,
        controller_statistics: dict[str, Any] | None = None,
        buffer_statistics: dict[str, Any] | None = None,
        configuration: dict[str, Any] | None = None,
        filename: str = "summary.json",
    ) -> Path:
        """
        Сохраняет сводную статистику обучения.
        """

        self._validate_history(history)

        if (
            controller_statistics is not None
            and not isinstance(
                controller_statistics,
                dict,
            )
        ):
            raise TypeError(
                "controller_statistics должен быть "
                "словарём или None."
            )

        if (
            buffer_statistics is not None
            and not isinstance(
                buffer_statistics,
                dict,
            )
        ):
            raise TypeError(
                "buffer_statistics должен быть "
                "словарём или None."
            )

        if (
            configuration is not None
            and not isinstance(
                configuration,
                dict,
            )
        ):
            raise TypeError(
                "configuration должен быть "
                "словарём или None."
            )

        path = self._prepare_path(filename)

        summary = {
            "training": history.summary(),
            "recent": {
                "success_rate_last_100": (
                    history.recent_success_rate(
                        window=100
                    )
                ),
                "average_reward_last_100": (
                    history.recent_average_reward(
                        window=100
                    )
                ),
            },
            "controller": (
                controller_statistics or {}
            ),
            "experience_buffer": (
                buffer_statistics or {}
            ),
            "configuration": (
                configuration or {}
            ),
        }

        self._write_json(
            path=path,
            data=summary,
        )

        return path

    def export_rules(
        self,
        rule_store: RuleStore,
        filename: str = "rules.json",
    ) -> Path:
        """
        Сохраняет правила из RuleStore.
        """

        self._validate_rule_store(rule_store)

        path = self._prepare_path(filename)

        self._write_json(
            path=path,
            data=rule_store.to_list(),
        )

        return path

    def export_q_table(
        self,
        q_policy: QTablePolicy,
        filename: str = "q_table.json",
    ) -> Path:
        """
        Сохраняет Q-таблицу и параметры политики.
        """

        self._validate_q_policy(q_policy)

        path = self._prepare_path(filename)

        self._write_json(
            path=path,
            data=q_policy.to_dict(),
        )

        return path

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
    def _write_json(
        path: Path,
        data: Any,
    ) -> None:
        with path.open(
            mode="w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
                default=TrainingExporter._json_default,
            )

    @staticmethod
    def _json_default(value: Any) -> Any:
        """
        Преобразует нестандартные объекты в JSON.

        Это полезно, если ключи состояний или другие
        значения представлены кортежами, множествами
        или объектами Path.
        """

        if isinstance(value, Path):
            return str(value)

        if isinstance(value, tuple):
            return list(value)

        if isinstance(value, set):
            return sorted(value)

        if hasattr(value, "to_dict"):
            return value.to_dict()

        return str(value)

    @staticmethod
    def _validate_history(
        history: TrainingHistory,
    ) -> None:
        if not isinstance(
            history,
            TrainingHistory,
        ):
            raise TypeError(
                "history должен быть объектом "
                "TrainingHistory."
            )

    @staticmethod
    def _validate_rule_store(
        rule_store: RuleStore,
    ) -> None:
        if not isinstance(
            rule_store,
            RuleStore,
        ):
            raise TypeError(
                "rule_store должен быть объектом "
                "RuleStore."
            )

    @staticmethod
    def _validate_q_policy(
        q_policy: QTablePolicy,
    ) -> None:
        if not isinstance(
            q_policy,
            QTablePolicy,
        ):
            raise TypeError(
                "q_policy должен быть объектом "
                "QTablePolicy."
            )