from dataclasses import dataclass
from typing import Any

Position = tuple[int, int]


@dataclass(frozen=True, slots=True)
class State:
    """
    Структурированное состояние среды MiniGrid.
    """

    agent_position: Position
    agent_direction: int

    key_position: Position | None = None
    door_position: Position | None = None
    goal_position: Position | None = None

    has_key: bool = False
    door_open: bool = False
    door_locked: bool = False

    key_visible: bool = False
    door_visible: bool = False
    goal_visible: bool = False

    def __post_init__(self) -> None:
        if self.agent_direction not in range(4):
            raise ValueError(
                "agent_direction должен быть числом от 0 до 3."
            )

    @property
    def agent_x(self) -> int:
        return self.agent_position[0]

    @property
    def agent_y(self) -> int:
        return self.agent_position[1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_x": self.agent_x,
            "agent_y": self.agent_y,
            "agent_direction": self.agent_direction,
            "key_position": self.key_position,
            "door_position": self.door_position,
            "goal_position": self.goal_position,
            "has_key": self.has_key,
            "door_open": self.door_open,
            "door_locked": self.door_locked,
            "key_visible": self.key_visible,
            "door_visible": self.door_visible,
            "goal_visible": self.goal_visible,
        }