from typing import Any

from src.state.state import Position, State


class StateParser:
    """
    Преобразует словарь признаков среды в объект State.
    """

    @staticmethod
    def parse(raw_state: dict[str, Any]) -> State:
        """
        Создаёт State из словаря.

        Ожидаемые поля:
        - agent_x
        - agent_y
        - agent_dir или agent_direction

        Остальные поля необязательны.
        """
        agent_x = StateParser._required_int(raw_state, "agent_x")
        agent_y = StateParser._required_int(raw_state, "agent_y")

        agent_direction = raw_state.get(
            "agent_direction",
            raw_state.get("agent_dir"),
        )

        if not isinstance(agent_direction, int):
            raise ValueError(
                "Состояние должно содержать целое поле "
                "'agent_direction' или 'agent_dir'."
            )

        return State(
            agent_position=(agent_x, agent_y),
            agent_direction=agent_direction,
            key_position=StateParser._read_position(raw_state, "key"),
            door_position=StateParser._read_position(raw_state, "door"),
            goal_position=StateParser._read_position(raw_state, "goal"),
            has_key=bool(raw_state.get("has_key", False)),
            door_open=bool(raw_state.get("door_open", False)),
            door_locked=bool(raw_state.get("door_locked", False)),
            key_visible=bool(raw_state.get("key_visible", False)),
            door_visible=bool(raw_state.get("door_visible", False)),
            goal_visible=bool(raw_state.get("goal_visible", False)),
        )

    @staticmethod
    def _required_int(
        raw_state: dict[str, Any],
        field_name: str,
    ) -> int:
        value = raw_state.get(field_name)

        if not isinstance(value, int):
            raise ValueError(
                f"Поле '{field_name}' должно быть целым числом."
            )

        return value

    @staticmethod
    def _read_position(
        raw_state: dict[str, Any],
        object_name: str,
    ) -> Position | None:
        x = raw_state.get(f"{object_name}_x")
        y = raw_state.get(f"{object_name}_y")

        if x is None and y is None:
            return None

        if not isinstance(x, int) or not isinstance(y, int):
            raise ValueError(
                f"Координаты объекта '{object_name}' "
                "должны быть целыми числами."
            )

        # В старых состояниях значение -1 означало:
        # объект не обнаружен или координата неизвестна.
        if x < 0 or y < 0:
            return None

        return x, y