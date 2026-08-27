from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.state.world_state import WorldState


class ObservationParser(ABC):
    """
    Базовый интерфейс преобразования наблюдения среды
    во внутреннее состояние агента.
    """

    @abstractmethod
    def parse(
        self,
        observation: Any,
        *,
        step: int = 0,
        info: dict[str, Any] | None = None,
    ) -> WorldState:
        """
        Преобразует наблюдение среды в WorldState.
        """
        raise NotImplementedError