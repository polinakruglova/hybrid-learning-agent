from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping


@dataclass(slots=True)
class WorldState:
    """
    Универсальное внутреннее состояние интеллектуального агента.

    attributes:
        Именованные признаки состояния.

    source:
        Название среды, из которой получено состояние.

    step:
        Номер шага внутри эпизода.

    metadata:
        Служебные данные, которые не должны напрямую участвовать
        в обучении или генерации правил.
    """

    attributes: dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
    step: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def get(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        return self.attributes.get(key, default)

    def set(
        self,
        key: str,
        value: Any,
    ) -> None:
        if not key:
            raise ValueError("State attribute name cannot be empty.")

        self.attributes[key] = value

    def update(
        self,
        values: Mapping[str, Any],
    ) -> None:
        self.attributes.update(values)

    def contains(
        self,
        key: str,
    ) -> bool:
        return key in self.attributes

    def remove(
        self,
        key: str,
    ) -> Any:
        return self.attributes.pop(key)

    def keys(self) -> Iterator[str]:
        return iter(self.attributes.keys())

    def values(self) -> Iterator[Any]:
        return iter(self.attributes.values())

    def items(self) -> Iterator[tuple[str, Any]]:
        return iter(self.attributes.items())

    def copy(self) -> "WorldState":
        return WorldState(
            attributes=dict(self.attributes),
            source=self.source,
            step=self.step,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "step": self.step,
            "attributes": dict(self.attributes),
            "metadata": dict(self.metadata),
        }

    def __getitem__(
        self,
        key: str,
    ) -> Any:
        return self.attributes[key]

    def __setitem__(
        self,
        key: str,
        value: Any,
    ) -> None:
        self.set(key, value)

    def __contains__(
        self,
        key: object,
    ) -> bool:
        return key in self.attributes

    def __len__(self) -> int:
        return len(self.attributes)

    def __repr__(self) -> str:
        preview_items = list(self.attributes.items())[:8]

        preview = ", ".join(
            f"{key}={value!r}"
            for key, value in preview_items
        )

        if len(self.attributes) > 8:
            preview += ", ..."

        return (
            f"WorldState("
            f"source={self.source!r}, "
            f"step={self.step}, "
            f"attributes={{"
            f"{preview}"
            f"}})"
        )