from __future__ import annotations

from dataclasses import asdict, is_dataclass
from numbers import Number
from typing import Any, Mapping, Sequence

import numpy as np

from src.state.observation_parser import ObservationParser
from src.state.world_state import WorldState


class UniversalObservationParser(ObservationParser):
    """
    Универсальный синтаксический анализатор наблюдений.

    Он преобразует произвольное наблюдение среды в плоский
    словарь признаков.

    Пример:

        {
            "agent": {
                "x": 3,
                "y": 5,
            },
            "health": 10,
        }

    становится:

        {
            "agent.x": 3,
            "agent.y": 5,
            "health": 10,
        }
    """

    def __init__(
        self,
        *,
        source_name: str = "generic",
        max_array_items: int = 256,
        include_array_statistics: bool = True,
        include_raw_arrays: bool = False,
    ) -> None:
        if max_array_items <= 0:
            raise ValueError(
                "max_array_items must be greater than zero."
            )

        self.source_name = source_name
        self.max_array_items = max_array_items
        self.include_array_statistics = include_array_statistics
        self.include_raw_arrays = include_raw_arrays

    def parse(
        self,
        observation: Any,
        *,
        step: int = 0,
        info: dict[str, Any] | None = None,
    ) -> WorldState:
        attributes: dict[str, Any] = {}

        self._flatten(
            value=observation,
            path="observation",
            output=attributes,
        )

        metadata = {
            "observation_type": type(observation).__name__,
        }

        if info is not None:
            metadata["info"] = dict(info)

        return WorldState(
            attributes=attributes,
            source=self.source_name,
            step=step,
            metadata=metadata,
        )

    def _flatten(
        self,
        *,
        value: Any,
        path: str,
        output: dict[str, Any],
    ) -> None:
        if value is None:
            output[path] = None
            return

        if isinstance(value, np.generic):
            output[path] = value.item()
            return

        if isinstance(value, (bool, str, bytes)):
            output[path] = self._normalise_scalar(value)
            return

        if isinstance(value, Number):
            output[path] = value
            return

        if is_dataclass(value) and not isinstance(value, type):
            self._flatten(
                value=asdict(value),
                path=path,
                output=output,
            )
            return

        if isinstance(value, Mapping):
            self._flatten_mapping(
                value=value,
                path=path,
                output=output,
            )
            return

        if isinstance(value, np.ndarray):
            self._flatten_array(
                value=value,
                path=path,
                output=output,
            )
            return

        if isinstance(value, Sequence):
            self._flatten_sequence(
                value=value,
                path=path,
                output=output,
            )
            return

        if hasattr(value, "__dict__"):
            self._flatten(
                value=vars(value),
                path=path,
                output=output,
            )
            return

        output[path] = repr(value)

    def _flatten_mapping(
        self,
        *,
        value: Mapping[Any, Any],
        path: str,
        output: dict[str, Any],
    ) -> None:
        if not value:
            output[f"{path}.empty"] = True
            return

        for key, item in value.items():
            safe_key = self._normalise_key(key)

            self._flatten(
                value=item,
                path=f"{path}.{safe_key}",
                output=output,
            )

    def _flatten_sequence(
        self,
        *,
        value: Sequence[Any],
        path: str,
        output: dict[str, Any],
    ) -> None:
        output[f"{path}.length"] = len(value)

        limit = min(
            len(value),
            self.max_array_items,
        )

        for index in range(limit):
            self._flatten(
                value=value[index],
                path=f"{path}.{index}",
                output=output,
            )

        if len(value) > limit:
            output[f"{path}.truncated"] = True
            output[f"{path}.original_length"] = len(value)

    def _flatten_array(
        self,
        *,
        value: np.ndarray,
        path: str,
        output: dict[str, Any],
    ) -> None:
        output[f"{path}.ndim"] = int(value.ndim)
        output[f"{path}.size"] = int(value.size)
        output[f"{path}.dtype"] = str(value.dtype)

        for dimension_index, dimension_size in enumerate(value.shape):
            output[
                f"{path}.shape.{dimension_index}"
            ] = int(dimension_size)

        if value.size == 0:
            output[f"{path}.empty"] = True
            return

        if self.include_array_statistics:
            self._add_array_statistics(
                value=value,
                path=path,
                output=output,
            )

        if self.include_raw_arrays:
            output[f"{path}.raw"] = value.copy()

        flattened = value.reshape(-1)
        limit = min(
            flattened.size,
            self.max_array_items,
        )

        for index in range(limit):
            item = flattened[index]

            if isinstance(item, np.generic):
                item = item.item()

            output[f"{path}.value.{index}"] = item

        if flattened.size > limit:
            output[f"{path}.truncated"] = True
            output[
                f"{path}.original_size"
            ] = int(flattened.size)

    @staticmethod
    def _add_array_statistics(
        *,
        value: np.ndarray,
        path: str,
        output: dict[str, Any],
    ) -> None:
        if not np.issubdtype(value.dtype, np.number):
            return

        numeric_value = value.astype(
            np.float64,
            copy=False,
        )

        finite_mask = np.isfinite(numeric_value)

        if not finite_mask.any():
            output[f"{path}.finite_values"] = 0
            return

        finite_values = numeric_value[finite_mask]

        output[f"{path}.minimum"] = float(
            finite_values.min()
        )
        output[f"{path}.maximum"] = float(
            finite_values.max()
        )
        output[f"{path}.mean"] = float(
            finite_values.mean()
        )
        output[f"{path}.standard_deviation"] = float(
            finite_values.std()
        )
        output[f"{path}.finite_values"] = int(
            finite_values.size
        )

    @staticmethod
    def _normalise_key(
        key: Any,
    ) -> str:
        text = str(key).strip()

        if not text:
            return "empty_key"

        return (
            text
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(".", "_")
        )

    @staticmethod
    def _normalise_scalar(
        value: bool | str | bytes,
    ) -> bool | str:
        if isinstance(value, bytes):
            return value.decode(
                "utf-8",
                errors="replace",
            )

        return value