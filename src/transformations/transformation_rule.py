from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TransformationRule:
    """
    Правило изменения мира, обнаруженное из опыта.
    """

    action: Any

    conditions: dict[str, Any] = field(
        default_factory=dict
    )

    effects: dict[str, tuple[str, Any]] = field(
        default_factory=dict
    )

    observations: int = 0

    confidence: float = 0.0

    def __str__(self) -> str:
        return (
            "TransformationRule("
            f"action={self.action}, "
            f"conditions={self.conditions}, "
            f"effects={self.effects}, "
            f"observations={self.observations}, "
            f"confidence={self.confidence:.2f}"
            ")"
        )