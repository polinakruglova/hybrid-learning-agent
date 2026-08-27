from __future__ import annotations

from typing import Any

import numpy as np

from src.state import WorldState


class MiniHackStateParser:
    """
    Преобразует сырое наблюдение MiniHack
    в осмысленное состояние WorldState.

    Первая версия извлекает:
    - координаты агента;
    - здоровье;
    - золото;
    - энергию;
    - сообщение среды;
    - размеры карты.
    """

    BLSTATS_INDEX = {
        "x": 0,
        "y": 1,
        "strength": 2,
        "strength_percent": 3,
        "dexterity": 4,
        "constitution": 5,
        "intelligence": 6,
        "wisdom": 7,
        "charisma": 8,
        "score": 9,
        "hitpoints": 10,
        "max_hitpoints": 11,
        "depth": 12,
        "gold": 13,
        "energy": 14,
        "max_energy": 15,
        "armor_class": 16,
        "monster_level": 17,
        "experience_level": 18,
        "experience_points": 19,
        "time": 20,
        "hunger": 21,
        "carrying_capacity": 22,
        "dungeon_number": 23,
        "level_number": 24,
        "condition": 25,
        "alignment": 26,
    }

    def parse(
        self,
        observation: dict[str, Any],
        *,
        step: int = 0,
        info: dict[str, Any] | None = None,
    ) -> WorldState:
        """
        Преобразует наблюдение MiniHack в WorldState.
        """
        if not isinstance(observation, dict):
            raise TypeError(
                "observation должен быть словарём."
            )

        blstats = observation.get("blstats")

        if not isinstance(blstats, np.ndarray):
            raise TypeError(
                "observation должен содержать массив blstats."
            )

        attributes = self._parse_blstats(blstats)

        attributes["health_ratio"] = self._safe_ratio(
            attributes["hitpoints"],
            attributes["max_hitpoints"],
        )

        attributes["energy_ratio"] = self._safe_ratio(
            attributes["energy"],
            attributes["max_energy"],
        )

        attributes["health_level"] = self._ratio_level(
            attributes["health_ratio"]
        )

        attributes["energy_level"] = self._ratio_level(
            attributes["energy_ratio"]
        )

        attributes["message"] = self._decode_message(
            observation.get("message")
        )

        glyphs = observation.get("glyphs")

        if isinstance(glyphs, np.ndarray) and glyphs.ndim == 2:
            attributes["map_height"] = int(glyphs.shape[0])
            attributes["map_width"] = int(glyphs.shape[1])
        else:
            attributes["map_height"] = 0
            attributes["map_width"] = 0

        metadata = {
            "observation_keys": tuple(observation.keys()),
            "info": dict(info or {}),
        }

        return WorldState(
            attributes=attributes,
            source="minihack",
            step=step,
            metadata=metadata,
        )

    def _parse_blstats(
        self,
        blstats: np.ndarray,
    ) -> dict[str, int]:
        """
        Преобразует массив blstats в именованные признаки.
        """
        if blstats.ndim != 1:
            raise ValueError(
                "blstats должен быть одномерным массивом."
            )

        required_size = max(
            self.BLSTATS_INDEX.values()
        ) + 1

        if blstats.size < required_size:
            raise ValueError(
                "В blstats недостаточно значений."
            )

        return {
            name: int(blstats[index])
            for name, index in self.BLSTATS_INDEX.items()
        }

    @staticmethod
    def _decode_message(
        message: Any,
    ) -> str:
        """
        Преобразует массив байтов MiniHack в строку.
        """
        if not isinstance(message, np.ndarray):
            return ""

        raw_bytes = bytes(
            int(value)
            for value in message
            if int(value) != 0
        )

        return raw_bytes.decode(
            "utf-8",
            errors="replace",
        ).strip()

    @staticmethod
    def _safe_ratio(
        value: int,
        maximum: int,
    ) -> float:
        if maximum <= 0:
            return 0.0

        return value / maximum

    @staticmethod
    def _ratio_level(
        ratio: float,
    ) -> str:
        if ratio <= 0.25:
            return "critical"

        if ratio <= 0.5:
            return "low"

        if ratio <= 0.75:
            return "medium"

        return "high"